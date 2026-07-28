import express from 'express';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import cors from 'cors';

const app = express();
app.use(cors());

const server = http.createServer(app);
const PORT = process.env.PORT || 8080;

// The AI Engine URL (FastAPI)
const ENGINE_WS_URL = 'ws://127.0.0.1:8000/ws/render';

const wss = new WebSocketServer({ server });

console.log(`🪐 Albedo Gateway Booting...`);

wss.on('connection', (clientWs) => {
  console.log('🟢 React Client Connected');

  // Open a dedicated proxy connection to the Python engine for this specific client
  // FIXED: Explicitly provide the Origin header to pass FastAPI's handshake validation
  const engineWs = new WebSocket(ENGINE_WS_URL, {
    headers: {
      'Origin': 'http://localhost:8080'
    }
  });

  engineWs.on('open', () => {
    console.log('🔥 Connected to Python AI Engine');
  });

  // When Python finishes the AI render, it sends a JPEG buffer back here.
  // We immediately forward this binary buffer to the React client.
  engineWs.on('message', (frameBuffer) => {
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(frameBuffer);
    }
  });

  // When the React client moves the mouse, it sends {x, y, z} JSON here.
  // We forward this JSON directly to the Python engine to trigger a new render.
  clientWs.on('message', (message) => {
    if (engineWs.readyState === WebSocket.OPEN) {
      engineWs.send(message.toString());
    }
  });

  clientWs.on('close', () => {
    console.log('🔴 React Client Disconnected');
    if (engineWs.readyState === WebSocket.OPEN) {
      engineWs.close();
    }
  });

  engineWs.on('close', () => {
    console.log('⚠️ Python AI Engine Disconnected');
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.close();
    }
  });

  engineWs.on('error', (err) => {
    console.error('AI Engine Error:', err.message);
  });
});

server.listen(PORT, () => {
  console.log(`🚀 Gateway Server listening on port ${PORT}`);
});