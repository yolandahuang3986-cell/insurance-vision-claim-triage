import assert from "node:assert/strict";
import { test } from "node:test";
import { spawn } from "node:child_process";

const port = 4179;

function startServer() {
  const child = spawn("npm", ["run", "start", "--", "--port", String(port)], { stdio: ["ignore", "pipe", "pipe"] });
  return child;
}

async function waitForServer() {
  for (let i = 0; i < 40; i += 1) {
    try {
      const response = await fetch(`http://localhost:${port}/`);
      if (response.ok) return response.text();
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Demo server did not start");
}

test("server renders the claim triage demo", async () => {
  const server = startServer();
  try {
    const html = await waitForServer();
    assert.match(html, /Insurance Vision/);
    assert.match(html, /Choose an intake/);
    assert.match(html, /Human-in-loop/i);
    assert.match(html, /Real model adapter reserved/i);
    assert.doesNotMatch(html, /codex-preview/);
  } finally {
    server.kill("SIGTERM");
  }
});

