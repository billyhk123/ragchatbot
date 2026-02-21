import { Client, LocalAuth } from "whatsapp-web.js";
import express from "express";
import QRCode from "qrcode";

const CHATBOT_URL = process.env.CHATBOT_URL || "http://localhost:8080";
const PORT = parseInt(process.env.BRIDGE_PORT || "3001", 10);

const app = express();
app.use(express.json());

let qrDataUrl = null; // base64 PNG of current QR code
let clientReady = false;
let clientInfo = null;

// ---------------------------------------------------------------------------
// WhatsApp Web client
// ---------------------------------------------------------------------------
const client = new Client({
  authStrategy: new LocalAuth({ dataPath: "./wa_session" }),
  puppeteer: {
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-gpu",
      "--disable-dev-shm-usage",
    ],
  },
});

client.on("qr", async (qr) => {
  clientReady = false;
  qrDataUrl = await QRCode.toDataURL(qr, { width: 320 });
  console.log("[WA] QR code generated – scan at http://localhost:" + PORT + "/qr");
});

client.on("ready", () => {
  clientReady = true;
  qrDataUrl = null;
  clientInfo = {
    pushname: client.info?.pushname,
    phone: client.info?.wid?.user,
  };
  console.log("[WA] Client ready –", clientInfo.pushname, clientInfo.phone);
});

client.on("authenticated", () => {
  console.log("[WA] Authenticated (session restored or QR scanned)");
});

client.on("auth_failure", (msg) => {
  console.error("[WA] Auth failure:", msg);
  qrDataUrl = null;
  clientReady = false;
});

client.on("disconnected", (reason) => {
  console.warn("[WA] Disconnected:", reason);
  clientReady = false;
  client.initialize().catch((e) => console.error("[WA] Re-init error:", e));
});

// Incoming message handler: forward to RAG chatbot, reply with answer
client.on("message", async (msg) => {
  if (msg.fromMe || msg.isStatus) return;
  if (msg.type !== "chat") return; // only plain text for now

  const sender = msg.from; // e.g. "628xxx@c.us"
  const text = msg.body;
  if (!text) return;

  console.log(`[WA] Message from ${sender}: ${text.slice(0, 80)}`);

  try {
    const res = await fetch(`${CHATBOT_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, user_id: sender }),
    });

    if (!res.ok) {
      console.error("[WA] Chatbot error:", res.status, await res.text());
      return;
    }

    const { answer } = await res.json();
    if (answer) {
      await msg.reply(answer);
      console.log(`[WA] Replied to ${sender}: ${answer.slice(0, 80)}`);
    }
  } catch (err) {
    console.error("[WA] Failed to get answer:", err.message);
  }
});

// ---------------------------------------------------------------------------
// HTTP endpoints
// ---------------------------------------------------------------------------

// QR code page (scan this with your phone)
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

// JSON status for health checks / monitoring
app.get("/status", (_req, res) => {
  res.json({
    connected: clientReady,
    info: clientInfo,
    qr_available: !!qrDataUrl,
  });
});

// Send a message programmatically (used by Python server if needed)
app.post("/send", async (req, res) => {
  if (!clientReady) {
    return res.status(503).json({ error: "WhatsApp client not connected" });
  }

  const { to, text } = req.body;
  if (!to || !text) {
    return res.status(400).json({ error: "Missing 'to' or 'text' in body" });
  }

  try {
    const chatId = to.includes("@") ? to : `${to}@c.us`;
    await client.sendMessage(chatId, text);
    res.json({ status: "sent", to: chatId });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Health
app.get("/health", (_req, res) => res.json({ status: "ok" }));

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`[Bridge] HTTP server on port ${PORT}`);
  console.log(`[Bridge] Chatbot URL: ${CHATBOT_URL}`);
});

client.initialize().catch((err) => {
  console.error("[WA] Failed to initialize:", err);
  process.exit(1);
});
