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

const MAX_CONV_TURNS = 5;
const MAX_CHARS_PER_TURN = 300;
const MAX_INTENT_CHARS = 500;
const SKIP_PREFIXES = ['<system-reminder>', '[Request interrupted by user]', 'Conversation info'];

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

/**
 * Extract text content from a message's content field.
 * Handles both string and array-of-blocks formats.
 */
function extractTextFromContent(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts = [];
  for (const block of content) {
    if (block.type === 'text' && block.text) {
      parts.push(block.text);
    }
  }
  return parts.join(' ');
}

/**
 * Check if text starts with any skip prefix.
 */
function shouldSkip(text) {
  for (const prefix of SKIP_PREFIXES) {
    if (text.startsWith(prefix)) return true;
  }
  return false;
}

/**
 * Extract recent conversation turns from BeforeToolCallContext.
 * Uses context.context.messages (AgentMessage[]) from pi-agent-core.
 */
function extractConversation(context) {
  try {
    const messages = context.context && context.context.messages;
    if (!Array.isArray(messages)) return [];

    const turns = [];
    for (let i = messages.length - 1; i >= 0 && turns.length < MAX_CONV_TURNS; i--) {
      const msg = messages[i];
      const role = msg.role;
      if (role !== 'user' && role !== 'assistant') continue;

      const text = extractTextFromContent(msg.content);
      if (!text || shouldSkip(text)) continue;

      turns.unshift({
        role: role,
        text: text.slice(0, MAX_CHARS_PER_TURN),
      });
    }
    return turns;
  } catch {
    return [];
  }
}

/**
 * Extract user intent (last user message text) from BeforeToolCallContext.
 */
function extractUserIntent(context) {
  try {
    const messages = context.context && context.context.messages;
    if (!Array.isArray(messages)) return '';

    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role !== 'user') continue;

      const text = extractTextFromContent(msg.content);
      if (text && !shouldSkip(text)) return text.slice(0, MAX_INTENT_CHARS);
    }
    return '';
  } catch {
    return '';
  }
}

function createCrossHook() {
  return async function crossBeforeToolCall(context, _signal) {
    const toolName =
      (context.toolCall && (context.toolCall.name || context.toolCall.type)) ||
      'unknown';
    const toolInput = context.args || {};

    // Extract conversation context and user intent from agent state
    const conversationContext = extractConversation(context);
    const userIntent = extractUserIntent(context);

    try {
      const result = await gateTool(toolName, toolInput, conversationContext, userIntent);
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

function gateTool(toolName, toolInput, conversationContext, userIntent) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      tool_name: toolName,
      tool_input: toolInput,
      agent: 'openclaw',
      session_id: SESSION_ID,
      conversation_context: conversationContext || [],
      user_intent: userIntent || '',
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
