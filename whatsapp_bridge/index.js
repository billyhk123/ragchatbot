import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  Browsers,
} from "@whiskeysockets/baileys";
import express from "express";
import QRCode from "qrcode";
import { Storage } from "@google-cloud/storage";
import { execSync } from "child_process";
import { existsSync, mkdirSync, rmSync } from "fs";

const CHATBOT_URL = process.env.CHATBOT_URL || "http://localhost:8080";
const PORT = parseInt(process.env.BRIDGE_PORT || "3001", 10);
const AUTH_DIR = process.env.AUTH_DIR || "./wa_session";
const GCS_SESSION_BUCKET = process.env.GCS_SESSION_BUCKET || "";
const GCS_SESSION_FILE = "wa_session.tar.gz";
const LOCAL_TAR = "/tmp/wa_session.tar.gz";
const BACKUP_INTERVAL_MS = 5 * 60 * 1000;

const app = express();
app.use(express.json());

let qrDataUrl = null;
let clientReady = false;
let clientInfo = null;
let sock = null;

// ---------------------------------------------------------------------------
// GCS session backup / restore
// ---------------------------------------------------------------------------
async function downloadSession() {
  if (!GCS_SESSION_BUCKET) return;
  try {
    const storage = new Storage();
    const file = storage.bucket(GCS_SESSION_BUCKET).file(GCS_SESSION_FILE);
    const [exists] = await file.exists();
    if (!exists) {
      console.log("[GCS] No session backup found – will need QR scan");
      return;
    }
    await file.download({ destination: LOCAL_TAR });
    mkdirSync(AUTH_DIR, { recursive: true });
    execSync(`tar -xzf ${LOCAL_TAR} -C ${AUTH_DIR}`);
    console.log("[GCS] Session restored from backup");
  } catch (err) {
    console.error("[GCS] Download failed:", err.message);
  }
}

async function uploadSession() {
  if (!GCS_SESSION_BUCKET) return;
  if (!existsSync(AUTH_DIR)) return;
  try {
    execSync(`tar -czf ${LOCAL_TAR} -C ${AUTH_DIR} .`);
    const storage = new Storage();
    await storage
      .bucket(GCS_SESSION_BUCKET)
      .upload(LOCAL_TAR, { destination: GCS_SESSION_FILE });
    console.log("[GCS] Session backed up");
  } catch (err) {
    console.error("[GCS] Upload failed:", err.message);
  }
}

// Periodic backup
setInterval(uploadSession, BACKUP_INTERVAL_MS);

// Graceful shutdown: save session before Cloud Run kills the container
process.on("SIGTERM", async () => {
  console.log("[Bridge] SIGTERM received – saving session");
  await uploadSession();
  process.exit(0);
});

// ---------------------------------------------------------------------------
// WhatsApp connection via Baileys (no Chromium needed)
// ---------------------------------------------------------------------------
let reconnectAttempts = 0;

async function startWhatsApp() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);

  let version;
  try {
    const fetched = await fetchLatestBaileysVersion();
    version = fetched.version;
    console.log("[WA] Using WA Web version:", version);
  } catch (e) {
    console.warn("[WA] Could not fetch latest version, using default:", e.message);
  }

  sock = makeWASocket({
    auth: state,
    browser: Browsers.ubuntu("Chrome"),
    ...(version && { version }),
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", async (update) => {
    const { connection, qr, lastDisconnect } = update;

    if (qr) {
      clientReady = false;
      reconnectAttempts = 0;
      qrDataUrl = await QRCode.toDataURL(qr, { width: 320 });
      console.log("[WA] QR code generated – scan at /qr");
    }

    if (connection === "open") {
      clientReady = true;
      reconnectAttempts = 0;
      qrDataUrl = null;
      clientInfo = {
        pushname: sock.user?.name || null,
        phone: sock.user?.id?.split(":")[0] || null,
      };
      console.log("[WA] Connected –", clientInfo.pushname, clientInfo.phone);
      await uploadSession();
    }

    if (connection === "close") {
      clientReady = false;
      const err = lastDisconnect?.error;
      const statusCode = err?.output?.statusCode;
      const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
      console.warn(
        "[WA] Disconnected – status:",
        statusCode,
        "message:", err?.message || "none",
        shouldReconnect ? "(reconnecting)" : "(logged out)"
      );

      if (statusCode === 405 || statusCode === 401 || statusCode === 403) {
        console.log("[WA] Clearing stale auth and restarting fresh");
        if (existsSync(AUTH_DIR)) {
          rmSync(AUTH_DIR, { recursive: true, force: true });
        }
        reconnectAttempts = 0;
      }

      if (shouldReconnect) {
        reconnectAttempts++;
        const delay = Math.min(3000 * reconnectAttempts, 30000);
        console.log(`[WA] Reconnecting in ${delay / 1000}s (attempt ${reconnectAttempts})`);
        setTimeout(() => startWhatsApp(), delay);
      } else {
        console.log("[WA] Logged out. Clearing auth for re-pair.");
        if (existsSync(AUTH_DIR)) {
          rmSync(AUTH_DIR, { recursive: true, force: true });
        }
      }
    }
  });

  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify") return;

    for (const msg of messages) {
      if (msg.key.fromMe) continue;

      const sender = msg.key.remoteJid;
      if (!sender || sender === "status@broadcast") continue;

      const text =
        msg.message?.conversation ||
        msg.message?.extendedTextMessage?.text ||
        "";
      if (!text) continue;

      console.log(`[WA] Message from ${sender}: ${text.slice(0, 80)}`);

      try {
        const res = await fetch(`${CHATBOT_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text, user_id: sender }),
        });

        if (!res.ok) {
          console.error("[WA] Chatbot error:", res.status, await res.text());
          continue;
        }

        const { answer } = await res.json();
        if (answer) {
          await sock.sendMessage(sender, { text: answer });
          console.log(`[WA] Replied to ${sender}: ${answer.slice(0, 80)}`);
        }
      } catch (err) {
        console.error("[WA] Failed to get answer:", err.message);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// HTTP endpoints
// ---------------------------------------------------------------------------
app.get("/qr", (_req, res) => {
  if (clientReady) {
    return res.send(
      `<html><body style="display:flex;align-items:center;justify-content:center;
       height:100vh;font-family:sans-serif;background:#0a1628;color:#25d366">
       <div style="text-align:center">
         <h1>&#x2705; WhatsApp Connected</h1>
         <p>${clientInfo?.pushname || ""} (${clientInfo?.phone || ""})</p>
       </div></body></html>`
    );
  }
  if (!qrDataUrl) {
    return res.send(
      `<html><head><meta http-equiv="refresh" content="3"></head>
       <body style="display:flex;align-items:center;justify-content:center;
       height:100vh;font-family:sans-serif;background:#0a1628;color:white">
       <div style="text-align:center">
         <h2>Waiting for QR code...</h2>
         <p>Page will auto-refresh.</p>
       </div></body></html>`
    );
  }
  res.send(
    `<html><head><meta http-equiv="refresh" content="20"></head>
     <body style="display:flex;align-items:center;justify-content:center;
     height:100vh;font-family:sans-serif;background:#0a1628;color:white">
     <div style="text-align:center">
       <h2>Scan with WhatsApp</h2>
       <img src="${qrDataUrl}" style="border-radius:12px"/>
       <p style="color:#aaa">Open WhatsApp &gt; Linked Devices &gt; Link a Device</p>
     </div></body></html>`
  );
});

app.get("/status", (_req, res) => {
  res.json({
    connected: clientReady,
    info: clientInfo,
    qr_available: !!qrDataUrl,
  });
});

app.post("/send", async (req, res) => {
  if (!clientReady || !sock) {
    return res.status(503).json({ error: "WhatsApp client not connected" });
  }

  const { to, text } = req.body;
  if (!to || !text) {
    return res.status(400).json({ error: "Missing 'to' or 'text' in body" });
  }

  try {
    const jid = to.includes("@") ? to : `${to}@s.whatsapp.net`;
    await sock.sendMessage(jid, { text });
    res.json({ status: "sent", to: jid });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/health", (_req, res) => res.json({ status: "ok" }));

// ---------------------------------------------------------------------------
// Start — restore session from GCS, then HTTP, then WhatsApp
// ---------------------------------------------------------------------------
async function main() {
  await downloadSession();

  app.listen(PORT, () => {
    console.log(`[Bridge] HTTP server on port ${PORT}`);
    console.log(`[Bridge] Chatbot URL: ${CHATBOT_URL}`);
    console.log(`[Bridge] GCS backup: ${GCS_SESSION_BUCKET || "disabled"}`);
    startWhatsApp().catch((err) =>
      console.error("[WA] Startup error:", err.message)
    );
  });
}

main();
