#!/usr/bin/env node
/**
 * Tour topology analyzer.
 * Usage: node ua-tour-analyze.js <input.json> <output.json>
 */
const fs = require('fs');

function main() {
  const [, , inputPath, outputPath] = process.argv;
  if (!inputPath || !outputPath) {
    console.error('Usage: node ua-tour-analyze.js <input.json> <output.json>');
    process.exit(1);
  }

  const raw = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = raw.nodes || [];
  const edges = raw.edges || [];
  const layers = raw.layers || [];

  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  // ---- Fan-in / fan-out ----
  const fanIn = new Map();
  const fanOut = new Map();
  for (const n of nodes) {
    fanIn.set(n.id, 0);
    fanOut.set(n.id, 0);
  }
  for (const e of edges) {
    if (fanOut.has(e.source) && fanIn.has(e.target)) {
      fanOut.set(e.source, fanOut.get(e.source) + 1);
      fanIn.set(e.target, fanIn.get(e.target) + 1);
    }
  }
  const fanInRanking = [...fanIn.entries()]
    .map(([id, count]) => ({ id, fanIn: count, name: (nodeById.get(id) || {}).name }))
    .sort((a, b) => b.fanIn - a.fanIn)
    .slice(0, 20);
  const fanOutRanking = [...fanOut.entries()]
    .map(([id, count]) => ({ id, fanOut: count, name: (nodeById.get(id) || {}).name }))
    .sort((a, b) => b.fanOut - a.fanOut)
    .slice(0, 20);

  // ---- Entry point candidates ----
  const ENTRY_NAMES = new Set([
    'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js',
    'server.ts', 'server.js', 'mod.rs', 'main.go', 'main.py', 'main.rs',
    'manage.py', 'app.py', 'wsgi.py', 'asgi.py', 'run.py', '__main__.py',
    'Application.java', 'Main.java', 'Program.cs', 'config.ru', 'index.php',
    'App.swift', 'Application.kt', 'main.cpp', 'main.c',
  ]);
  const fanOutValues = [...fanOut.values()].sort((a, b) => b - a);
  const fanOutP90 = fanOutValues[Math.floor(fanOutValues.length * 0.1)] || 0;
  const fanInValues = [...fanIn.values()].sort((a, b) => a - b);
  const fanInP25 = fanInValues[Math.floor(fanInValues.length * 0.25)] || 0;

  const candidates = [];
  for (const n of nodes) {
    let score = 0;
    const fp = n.filePath || '';
    const depth = fp.split('/').length; // 1 = root, 2 = one level deep
    const isDoc = n.type === 'document';
    const isCode = n.type === 'file' || n.type === 'config';

    if (isDoc) {
      if (n.name === 'README.md' && depth === 1) score += 5;
      else if (/\.md$/.test(n.name) && depth === 1) score += 2;
      else if (/\.md$/.test(n.name)) score += 1; // docs anywhere: mild signal
    }
    if (isCode) {
      if (ENTRY_NAMES.has(n.name)) score += 3;
      // training/inference launchers for research repos
      if (/^(train|infer|test)_.+\.py$/.test(n.name)) score += 2;
      if (depth <= 2) score += 1;
      if (fanOut.get(n.id) >= fanOutP90 && fanOut.get(n.id) > 0) score += 1;
      if (fanIn.get(n.id) <= fanInP25) score += 1;
    }
    if (score > 0) candidates.push({ id: n.id, score, name: n.name, type: n.type, summary: n.summary || '' });
  }
  candidates.sort((a, b) => b.score - a.score);
  const entryPointCandidates = candidates.slice(0, 8);

  // ---- BFS from top CODE entry candidate ----
  const adj = new Map();
  for (const e of edges) {
    if (e.type !== 'imports' && e.type !== 'calls' && e.type !== 'depends_on') continue;
    if (!adj.has(e.source)) adj.set(e.source, []);
    adj.get(e.source).push(e.target);
  }
  // Fallback: if no imports/calls/depends_on edges exist at all, use exports/contains/configures
  if (adj.size === 0) {
    for (const e of edges) {
      if (!adj.has(e.source)) adj.set(e.source, []);
      adj.get(e.source).push(e.target);
    }
  }

  const startNode =
    (entryPointCandidates.find((c) => c.type !== 'document') || {}).id || null;

  const bfsTraversal = { startNode, order: [], depthMap: {}, byDepth: {} };
  if (startNode) {
    const visited = new Set([startNode]);
    let queue = [startNode];
    let depth = 0;
    while (queue.length) {
      bfsTraversal.byDepth[String(depth)] = [];
      const next = [];
      for (const id of queue) {
        bfsTraversal.order.push(id);
        bfsTraversal.depthMap[id] = depth;
        bfsTraversal.byDepth[String(depth)].push(id);
        for (const t of adj.get(id) || []) {
          if (nodeById.has(t) && !visited.has(t)) {
            visited.add(t);
            next.push(t);
          }
        }
      }
      queue = next;
      depth += 1;
    }
  }

  // ---- Non-code inventory ----
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const n of nodes) {
    const entry = { id: n.id, name: n.name, summary: n.summary || '' };
    if (n.type === 'document') nonCodeFiles.documentation.push(entry);
    else if (['service', 'pipeline', 'resource'].includes(n.type)) nonCodeFiles.infrastructure.push(entry);
    else if (['table', 'schema', 'endpoint'].includes(n.type)) nonCodeFiles.data.push(entry);
    else if (n.type === 'config') nonCodeFiles.config.push(entry);
  }

  // ---- Tightly coupled clusters (file-level only) ----
  const pairKey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  const edgeCountBetween = new Map();
  const undirected = new Set();
  const directed = new Set();
  for (const e of edges) {
    if (!nodeById.has(e.source) || !nodeById.has(e.target)) continue;
    directed.add(`${e.source}->${e.target}`);
    undirected.add(pairKey(e.source, e.target));
    edgeCountBetween.set(pairKey(e.source, e.target), (edgeCountBetween.get(pairKey(e.source, e.target)) || 0) + 1);
  }

  const clusters = [];
  const clustered = new Set();
  // Seed with bidirectional or multi-edge pairs
  const seeds = [];
  for (const [pk, cnt] of edgeCountBetween) {
    const [a, b] = pk.split('|');
    const bidir = directed.has(`${a}->${b}`) && directed.has(`${b}->${a}`);
    if (bidir || cnt >= 2) seeds.push({ a, b, cnt });
  }
  seeds.sort((x, y) => y.cnt - x.cnt);
  for (const s of seeds) {
    if (clustered.has(s.a) || clustered.has(s.b)) continue;
    const cluster = new Set([s.a, s.b]);
    // Expand: add nodes connecting to 2+ cluster members
    let grown = true;
    while (grown && cluster.size < 5) {
      grown = false;
      for (const n of nodes) {
        if (cluster.has(n.id) || clustered.has(n.id)) continue;
        let links = 0;
        for (const m of cluster) if (undirected.has(pairKey(n.id, m))) links++;
        if (links >= 2) {
          cluster.add(n.id);
          grown = true;
        }
      }
    }
    let edgeCount = 0;
    const arr = [...cluster];
    for (let i = 0; i < arr.length; i++)
      for (let j = i + 1; j < arr.length; j++)
        edgeCount += edgeCountBetween.get(pairKey(arr[i], arr[j])) || 0;
    clusters.push({ nodes: arr, edgeCount });
    for (const m of cluster) clustered.add(m);
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount);

  // ---- Layers ----
  const layersOut = {
    count: layers.length,
    list: layers.map((l) => ({ id: l.id, name: l.name, description: l.description, nodeIds: l.nodeIds || [] })),
  };

  // ---- Node summary index ----
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary || '', filePath: n.filePath || '' };
  }

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal,
    nonCodeFiles,
    clusters: clusters.slice(0, 10),
    layers: layersOut,
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
  };

  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  console.log('OK: wrote', outputPath);
}

try {
  main();
} catch (err) {
  console.error('FATAL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
