/**
 * cross OpenClaw hook (ESM entry point)
 *
 * Loaded via NODE_OPTIONS="--import /path/to/openclaw_hook.mjs"
 *
 * Since OpenClaw uses ESM imports, we can't use Module._load to intercept.
 * Instead, we poll for the Agent class after modules have loaded and patch
 * its prototype to install our beforeToolCall hook.
 *
 * The hook POSTs to the cross daemon's /cross/api/gate endpoint for gate
 * evaluation. Fails open (allows) if the daemon is unreachable.
 */

import http from 'node:http';
import { createRequire } from 'node:module';

const CROSS_PORT = process.env.CROSS_LISTEN_PORT || '2767';
const SESSION_ID = process.env.CROSS_SESSION_ID || '';
const TIMEOUT_MS = 300000; // 5 minutes — allows for human escalation review

let patched = false;

/**
 * Try to find and patch the Agent class from pi-agent-core.
 * Uses createRequire to resolve the CJS entry point of the package,
 * which re-exports the ESM module's symbols.
 */
async function tryPatch() {
  if (patched) return true;

  // pi-agent-core is ESM — must use dynamic import()
  const candidates = [
    // OpenClaw Homebrew install
    'file:///opt/homebrew/lib/node_modules/openclaw/node_modules/@mariozechner/pi-agent-core/dist/agent.js',
    // Alternative global install
    'file:///usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-agent-core/dist/agent.js',
  ];

  // Try resolving from the script being run (process.argv[1])
  if (process.argv[1]) {
    try {
      const scriptRequire = createRequire(process.argv[1]);
      const resolved = scriptRequire.resolve('@mariozechner/pi-agent-core/dist/agent.js');
      candidates.unshift('file://' + resolved);
    } catch {
      // Not resolvable from script path
    }
  }

  for (const url of candidates) {
    try {
      const mod = await import(url);
      if (mod && mod.Agent && typeof mod.Agent === 'function') {
        const AgentProto = mod.Agent.prototype;
        if (typeof AgentProto.setBeforeToolCall === 'function') {
          installCrossHook(AgentProto);
          patched = true;
          return true;
        }
      }
    } catch {
      // Module not found at this path
    }
  }

  return false;
}

function installCrossHook(AgentProto) {
  const originalSetBefore = AgentProto.setBeforeToolCall;

  // Override setBeforeToolCall to chain our hook before any user-set hook
  AgentProto.setBeforeToolCall = function (userHook) {
    const crossHook = createCrossHook();

    const chainedHook = async (context, signal) => {
      const crossResult = await crossHook(context, signal);
      if (crossResult && crossResult.block) {
        return crossResult;
      }
      if (userHook) {
        return userHook(context, signal);
      }
      return { block: false };
    };

    return originalSetBefore.call(this, chainedHook);
  };

  // Install hook on first prompt() call for agents that don't call setBeforeToolCall
  const originalPrompt = AgentProto.prompt;
  if (originalPrompt) {
    AgentProto.prompt = function (...args) {
      if (!this._crossHookInstalled) {
        this._crossHookInstalled = true;
        console.error('[cross] Installing hook via prompt() interception');
        const existing = this._beforeToolCall;
        const crossHook = createCrossHook();

        if (existing) {
          const chainedHook = async (ctx, sig) => {
            const r = await crossHook(ctx, sig);
            if (r && r.block) return r;
            return existing(ctx, sig);
          };
          originalSetBefore.call(this, chainedHook);
        } else {
          originalSetBefore.call(this, crossHook);
        }
      }
      return originalPrompt.apply(this, args);
    };
  }

  console.error('[cross] OpenClaw tool hook installed');
}

function createCrossHook() {
  return async function crossBeforeToolCall(context, _signal) {
    const toolName =
      (context.toolCall && (context.toolCall.name || context.toolCall.type)) ||
      'unknown';
    const toolInput = context.args || {};

    try {
      const result = await gateTool(toolName, toolInput);
      if (result.action === 'BLOCK' || result.action === 'HALT_SESSION') {
        return {
          block: true,
          reason: '[cross] ' + (result.reason || 'Blocked by cross'),
        };
      }
      if (result.action === 'ESCALATE') {
        return {
          block: true,
          reason:
            '[cross] ' + (result.reason || 'Escalated to human review'),
        };
      }
      return { block: false };
    } catch (err) {
      console.error('[cross] Gate check failed (allowing): ' + err.message);
      return { block: false };
    }
  };
}

function gateTool(toolName, toolInput) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      tool_name: toolName,
      tool_input: toolInput,
      agent: 'openclaw',
      session_id: SESSION_ID,
    });

    const req = http.request(
      {
        hostname: 'localhost',
        port: parseInt(CROSS_PORT, 10),
        path: '/cross/api/gate',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
        timeout: TIMEOUT_MS,
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(new Error('Invalid response: ' + data));
          }
        });
      }
    );

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Gate request timed out'));
    });

    req.write(payload);
    req.end();
  });
}

// Patch immediately (top-level await)
if (!(await tryPatch())) {
  // Retry after short delay (modules may still be loading)
  await new Promise(r => setTimeout(r, 100));
  if (!(await tryPatch())) {
    console.error('[cross] Could not find pi-agent-core Agent class — hook not installed');
    console.error('[cross] Will retry on first Agent.prompt() call if class loads later');
  }
}

// Fallback: periodically check for unpatched Agent instances
// This catches cases where the Agent is created after our initial patch
const _patchTimer = setInterval(async () => {
  if (patched) {
    clearInterval(_patchTimer);
  } else {
    await tryPatch();
    if (patched) clearInterval(_patchTimer);
  }
}, 5000);
