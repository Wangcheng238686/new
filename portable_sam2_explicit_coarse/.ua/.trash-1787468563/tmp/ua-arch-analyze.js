#!/usr/bin/env node
'use strict';
const fs = require('fs');

function main() {
  const [, , inPath, outPath] = process.argv;
  if (!inPath || !outPath) {
    console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
    process.exit(1);
  }
  const input = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const fileNodes = input.fileNodes || [];
  const importEdges = input.importEdges || [];
  const allEdges = input.allEdges || [];

  const nodeById = new Map(fileNodes.map(n => [n.id, n]));

  // ---------- A. Directory grouping ----------
  // compute common path prefix across all file paths
  function splitPath(p) { return p.split('/'); }
  let prefixParts = [];
  if (fileNodes.length > 0) {
    prefixParts = splitPath(fileNodes[0].filePath);
    for (const n of fileNodes) {
      const parts = splitPath(n.filePath);
      let i = 0;
      while (i < prefixParts.length && i < parts.length && prefixParts[i] === parts[i] && parts[i] !== '') i++;
      prefixParts = prefixParts.slice(0, i);
    }
  }
  // prefix only counts directory segments (drop the filename portion)
  const prefix = prefixParts.length ? prefixParts.join('/') + '/' : '';
  const firstSegOf = {};
  for (const n of fileNodes) {
    const rel = prefix ? n.filePath.slice(prefix.length) : n.filePath;
    const segs = rel.split('/');
    const group = segs.length > 1 ? segs[0] : 'root';
    firstSegOf[n.id] = group;
  }

  // Deeper second-level grouping for groups with many files (scripts/ablations)
  function groupOf(n) {
    const g = firstSegOf[n.id];
    if (g === 'scripts' && n.filePath.includes('scripts/ablations/')) {
      return 'scripts/ablations';
    }
    return g;
  }

  const directoryGroups = {};
  for (const n of fileNodes) {
    const g = groupOf(n);
    (directoryGroups[g] ||= []).push(n.id);
  }

  // ---------- B. Node type grouping ----------
  const nodeTypeGroups = {};
  for (const n of fileNodes) {
    (nodeTypeGroups[n.type] ||= []).push(n.id);
  }

  // ---------- C. Import adjacency: fan-in / fan-out ----------
  const fileFanIn = {};
  const fileFanOut = {};
  for (const e of importEdges) {
    fileFanOut[e.source] = (fileFanOut[e.source] || 0) + 1;
    fileFanIn[e.target] = (fileFanIn[e.target] || 0) + 1;
  }

  // ---------- D. Cross-category dependency analysis ----------
  const crossCat = {};
  for (const e of allEdges) {
    const s = nodeById.get(e.source);
    const t = nodeById.get(e.target);
    if (!s || !t) continue;
    const key = `${s.type}|${t.type}|${e.type}`;
    crossCat[key] = (crossCat[key] || 0) + 1;
  }
  const crossCategoryEdges = Object.entries(crossCat).map(([k, count]) => {
    const [fromType, toType, edgeType] = k.split('|');
    return { fromType, toType, edgeType, count };
  }).sort((a, b) => b.count - a.count);

  // ---------- E. Inter-group import frequency (import edges + depends_on) ----------
  const interPair = {};
  for (const e of [...importEdges, ...allEdges.filter(x => x.type === 'depends_on')]) {
    const s = nodeById.get(e.source);
    const t = nodeById.get(e.target);
    if (!s || !t) continue;
    const gs = groupOf(s), gt = groupOf(t);
    if (gs === gt) continue;
    const key = `${gs}|${gt}|${e.type}`;
    interPair[key] = (interPair[key] || 0) + 1;
  }
  const interGroupImports = Object.entries(interPair).map(([k, count]) => {
    const [from, to, type] = k.split('|');
    return { from, to, type, count };
  }).sort((a, b) => b.count - a.count);

  // ---------- F. Intra-group density ----------
  const intraGroupDensity = {};
  for (const g of Object.keys(directoryGroups)) intraGroupDensity[g] = { internalEdges: 0, totalEdges: 0, density: 0 };
  for (const e of allEdges) {
    const s = nodeById.get(e.source);
    const t = nodeById.get(e.target);
    if (!s || !t) continue;
    const gs = groupOf(s), gt = groupOf(t);
    intraGroupDensity[gs].totalEdges += 1;
    if (gt === gs) {
      intraGroupDensity[gs].internalEdges += 1;
    } else {
      intraGroupDensity[gt].totalEdges += 1;
    }
  }
  for (const g of Object.keys(intraGroupDensity)) {
    const d = intraGroupDensity[g];
    d.density = d.totalEdges > 0 ? +(d.internalEdges / d.totalEdges).toFixed(2) : 0;
  }

  // ---------- G. Directory pattern matching ----------
  const DIR_PATTERNS = [
    [/^(routes|api|controllers?|endpoints|handlers|serializers|routers|blueprints)$/, 'api'],
    [/^(services|core|lib|domain|logic|signals|internal|composables|mailers|jobs|channels)$/, 'service'],
    [/^(models|db|data|persistence|repository|entities|entity|migrations|sql|database|schema)$/, 'data'],
    [/^(components|views|pages|ui|layouts|screens)$/, 'ui'],
    [/^(middleware|plugins|interceptors|guards)$/, 'middleware'],
    [/^(utils|helpers|common|shared|tools|pkg|templatetags)$/, 'utility'],
    [/^(config|configs|constants|env|settings|management|commands)$/, 'config'],
    [/^(__tests__|tests?|specs?)$/, 'test'],
    [/^(types|interfaces|schemas|contracts|dtos?|request|response)$/, 'types'],
    [/^hooks$/, 'hooks'],
    [/^(store|state|reducers|actions|slices)$/, 'state'],
    [/^(assets|static|public)$/, 'assets'],
    [/^(cmd|bin)$/, 'entry'],
    [/^(docs|documentation|wiki)$/, 'documentation'],
    [/^(deploy|deployment|infra|infrastructure|k8s|kubernetes|helm|charts|terraform|tf|docker)$/, 'infrastructure'],
    [/^(\.github|\.gitlab|\.circleci)$/, 'ci-cd'],
    [/^(train|training)$/, 'entry'],
    [/^(inference|infer)$/, 'entry'],
    [/^(scripts|script)$/, 'utility'],
    [/^rsprompter$/, 'service'],
  ];
  const patternMatches = {};
  for (const g of Object.keys(directoryGroups)) {
    // check last segment of the group name too (e.g. scripts/ablations)
    const segs = g.split('/');
    let label = null;
    for (const seg of [...segs].reverse()) {
      for (const [re, lab] of DIR_PATTERNS) {
        if (re.test(seg)) { label = lab; break; }
      }
      if (label) break;
    }
    patternMatches[g] = label || null;
  }

  // File-level pattern matching
  const filePatterns = {};
  for (const n of fileNodes) {
    const p = n.filePath;
    const base = p.split('/').pop();
    let label = null;
    if (/(\.test\.|\.spec\.|^test_.*\.py$|_test\.go$|Test\.java$|_spec\.rb$|Test\.php$|Tests\.cs$)/.test(base)) label = 'test';
    else if (/\.d\.ts$/.test(base)) label = 'types';
    else if (base === '__init__.py') label = 'entry';
    else if (base === 'manage.py') label = 'entry';
    else if (/^(wsgi|asgi)\.py$/.test(base)) label = 'config';
    else if (/^main\.go$/.test(base) && /cmd\//.test(p)) label = 'entry';
    else if (/^(main|lib)\.rs$/.test(base) && p.startsWith('src/')) label = 'entry';
    else if (/(Application\.java|Program\.cs)$/.test(base)) label = 'entry';
    else if (base === 'config.ru') label = 'entry';
    else if (/^(Cargo\.toml|go\.mod|Gemfile|pom\.xml|build\.gradle|composer\.json)$/.test(base)) label = 'config';
    else if (/^Dockerfile/.test(base) || /^docker-compose\./.test(base)) label = 'infrastructure';
    else if (/\.tf(vars)?$/.test(base)) label = 'infrastructure';
    else if (/\.github\/workflows\//.test(p) || base === '.gitlab-ci.yml' || base === 'Jenkinsfile') label = 'ci-cd';
    else if (/\.sql$/.test(base)) label = 'data';
    else if (/\.(graphql|gql|proto)$/.test(base)) label = 'types';
    else if (/\.(md|rst)$/.test(base)) label = 'documentation';
    else if (base === 'Makefile') label = 'infrastructure';
    else if (/\.sh$/.test(base)) label = 'script';
    else if (/\.py$/.test(base)) label = 'python';
    else if (/\.json$/.test(base)) label = 'config';
    if (label) filePatterns[n.id] = label;
  }

  // ---------- H. Deployment topology ----------
  const infraFiles = [];
  let hasDockerfile = false, hasCompose = false, hasK8s = false, hasTerraform = false, hasCI = false;
  for (const n of fileNodes) {
    const base = n.filePath.split('/').pop();
    if (/^Dockerfile/.test(base)) { hasDockerfile = true; infraFiles.push(n.filePath); }
    else if (/^docker-compose\./.test(base)) { hasCompose = true; infraFiles.push(n.filePath); }
    else if (/(^|\/)(k8s|kubernetes|helm|charts)\//.test(n.filePath)) { hasK8s = true; infraFiles.push(n.filePath); }
    else if (/\.tf(vars)?$/.test(base)) { hasTerraform = true; infraFiles.push(n.filePath); }
    else if (/\.github\/workflows\//.test(n.filePath) || base === '.gitlab-ci.yml' || base === 'Jenkinsfile') { hasCI = true; infraFiles.push(n.filePath); }
  }
  const deploymentTopology = { hasDockerfile, hasCompose, hasK8s, hasTerraform, hasCI, infraFiles };

  // ---------- I. Data pipeline detection ----------
  const schemaFiles = [], migrationFiles = [], dataModelFiles = [], apiHandlerFiles = [];
  for (const n of fileNodes) {
    const t = (n.tags || []).join(' ').toLowerCase();
    if (/\.(sql|graphql|gql|proto)$/.test(n.filePath) || n.type === 'schema') schemaFiles.push(n.filePath);
    if (/migrations?\//.test(n.filePath)) migrationFiles.push(n.filePath);
    if (/data-model|dataset|coco|preprocessing|data-pipeline/.test(t)) dataModelFiles.push(n.filePath);
    if (/api-handler|endpoint|inference/.test(t)) apiHandlerFiles.push(n.filePath);
  }
  const dataPipeline = { schemaFiles, migrationFiles, dataModelFiles, apiHandlerFiles };

  // ---------- J. Documentation coverage ----------
  const docPaths = fileNodes.filter(n => n.type === 'document' || /\.(md|rst)$/.test(n.filePath)).map(n => n.filePath);
  const groupsWithDocsSet = new Set();
  for (const dp of docPaths) {
    const g = groupOf({ id: 'x', filePath: dp });
    groupsWithDocsSet.add(g);
  }
  const totalGroups = Object.keys(directoryGroups).length;
  const undocumentedGroups = Object.keys(directoryGroups).filter(g => !groupsWithDocsSet.has(g));
  const docCoverage = {
    groupsWithDocs: groupsWithDocsSet.size,
    totalGroups,
    coverageRatio: totalGroups ? +(groupsWithDocsSet.size / totalGroups).toFixed(2) : 0,
    undocumentedGroups,
  };

  // ---------- K. Dependency direction ----------
  const pairNet = {};
  for (const { from, to, count } of interGroupImports) {
    const key = from < to ? `${from}|${to}` : `${to}|${from}`;
    const sign = from < to ? count : -count;
    pairNet[key] = (pairNet[key] || 0) + sign;
  }
  const dependencyDirection = [];
  for (const [key, net] of Object.entries(pairNet)) {
    const [a, b] = key.split('|');
    if (net > 0) dependencyDirection.push({ dependent: a, dependsOn: b, weight: net });
    else if (net < 0) dependencyDirection.push({ dependent: b, dependsOn: a, weight: -net });
  }
  dependencyDirection.sort((x, y) => y.weight - x.weight);

  // ---------- stats ----------
  const filesPerGroup = {};
  for (const [g, ids] of Object.entries(directoryGroups)) filesPerGroup[g] = ids.length;
  const nodeTypeCounts = {};
  for (const [t, ids] of Object.entries(nodeTypeGroups)) nodeTypeCounts[t] = ids.length;

  const result = {
    scriptCompleted: true,
    commonPrefix: prefix,
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    filePatterns,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup,
      nodeTypeCounts,
    },
    fileFanIn,
    fileFanOut,
  };
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.log('OK: wrote', outPath);
}

try { main(); } catch (err) {
  console.error('FATAL:', err && err.stack || err);
  process.exit(1);
}
