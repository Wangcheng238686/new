#!/usr/bin/env node
import { spawn } from "node:child_process";

const ZCODE = "/home/wangcheng/.npm-global/bin/zcode";
const MAX_TASK_LENGTH = 120_000;
const MAX_CWD_LENGTH = 512;

function send(id, result, error) {
  const body = JSON.stringify(error ? { jsonrpc: "2.0", id, error } : { jsonrpc: "2.0", id, result });
  process.stdout.write(`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
}

function toolSchema() {
  return {
    name: "zcode_implement",
    description: "Delegate an implementation task to the authenticated ZCode CLI in the selected project directory.",
    inputSchema: {
      type: "object",
      additionalProperties: false,
      required: ["task", "cwd"],
      properties: {
        task: { type: "string", minLength: 1, maxLength: MAX_TASK_LENGTH },
        cwd: { type: "string", minLength: 1, maxLength: MAX_CWD_LENGTH },
        mode: { type: "string", enum: ["build", "edit", "plan", "yolo"], default: "build" }
      }
    }
  };
}

function runZcode({ task, cwd, mode }) {
  return new Promise((resolve) => {
    const child = spawn(ZCODE, ["--cwd", cwd, "--mode", mode || "build", "--no-color", "--prompt", task], {
      cwd,
      env: { ...process.env },
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout = "", stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => resolve({ code: -1, stdout, stderr: String(error) }));
    child.on("close", (code, signal) => resolve({ code: code ?? -1, signal, stdout, stderr }));
  });
}

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.resume();
process.stdin.on("data", (chunk) => {
  input += chunk;
  let newline;
  while ((newline = input.indexOf("\n")) >= 0) {
    const raw = input.slice(0, newline).trim();
    input = input.slice(newline + 1);
    if (!raw) continue;
    try { handle(JSON.parse(raw)).catch((error) => send(null, null, { code: -32603, message: String(error) })); }
    catch (error) { send(null, null, { code: -32700, message: String(error) }); }
  }
});

async function handle(message) {
  const { id, method, params = {} } = message;
  if (method === "initialize") {
    return send(id, { protocolVersion: "2024-11-05", capabilities: { tools: {} }, serverInfo: { name: "zcode-mcp-bridge", version: "0.1.0" } });
  }
  if (method === "notifications/initialized") return;
  if (method === "tools/list") return send(id, { tools: [toolSchema()] });
  if (method === "tools/call") {
    if (params.name !== "zcode_implement") return send(id, null, { code: -32601, message: `Unknown tool: ${params.name}` });
    const args = params.arguments || {};
    if (typeof args.task !== "string" || typeof args.cwd !== "string") return send(id, null, { code: -32602, message: "task and cwd are required" });
    if (args.task.length > MAX_TASK_LENGTH || args.cwd.length > MAX_CWD_LENGTH) return send(id, null, { code: -32602, message: "task or cwd is too long" });
    const result = await runZcode(args);
    const text = [result.stdout, result.stderr ? `\n[stderr]\n${result.stderr}` : ""].join("").trim() || `(zcode exited with code ${result.code})`;
    return send(id, { content: [{ type: "text", text }], isError: result.code !== 0 });
  }
  if (id !== undefined) send(id, null, { code: -32601, message: `Unsupported method: ${method}` });
}
