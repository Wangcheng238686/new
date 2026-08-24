const fs = require('fs');
const graphPath = '/home/wangcheng/project/new/portable_sam2_explicit_coarse/.ua/intermediate/assembled-graph.json';
const g = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
const nodes = g.nodes || [];
const edges = g.edges || [];
const FILE_TYPES = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
const fileNodes = nodes.filter(n => FILE_TYPES.has(n.type)).map(n => ({
  id: n.id, type: n.type, name: n.name, filePath: n.filePath, summary: n.summary || '', tags: n.tags || []
}));
const fileIds = new Set(fileNodes.map(n => n.id));
const importEdges = edges.filter(e => e.type === 'imports' && fileIds.has(e.source) && fileIds.has(e.target))
  .map(e => ({ source: e.source, target: e.target, type: e.type }));
const allEdges = edges.filter(e => fileIds.has(e.source) && fileIds.has(e.target))
  .map(e => ({ source: e.source, target: e.target, type: e.type }));
const out = { fileNodes, importEdges, allEdges };
fs.writeFileSync('/home/wangcheng/project/new/portable_sam2_explicit_coarse/.ua/tmp/ua-arch-input.json', JSON.stringify(out, null, 2));
console.log('fileNodes:', fileNodes.length, 'importEdges:', importEdges.length, 'allEdges:', allEdges.length);
const byType = {};
fileNodes.forEach(n => byType[n.type] = (byType[n.type] || 0) + 1);
console.log('byType:', JSON.stringify(byType));
const edgeTypes = {};
allEdges.forEach(e => edgeTypes[e.type] = (edgeTypes[e.type] || 0) + 1);
console.log('edgeTypes:', JSON.stringify(edgeTypes));
