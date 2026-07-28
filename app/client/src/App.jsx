import React, { useEffect, useRef, useState } from 'react';

// The URL of our Node.js WebSocket Proxy
const GATEWAY_WS_URL = 'ws://localhost:8080';

export default function App() {
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const [isConnected, setIsConnected] = useState(false);
    const [fps, setFps] = useState(0);

    // Refs for calculating FPS without causing re-renders
    const frameCountRef = useRef(0);
    const lastTimeRef = useRef(performance.now());

    useEffect(() => {
        console.log("Attempting connection to Albedo Gateway...");
        
        try {
            const ws = new WebSocket(GATEWAY_WS_URL);
            wsRef.current = ws;
            
            // We expect raw binary data (the JPEG buffer) from the server
            ws.binaryType = "blob"; 

            ws.onopen = () => {
                console.log("🟢 Connected to Albedo Gateway!");
                setIsConnected(true);
            };

            // ADD THIS NEW BLOCK: Handle incoming binary frames
            ws.onmessage = async (event) => {
                const canvas = canvasRef.current;
                if (!canvas) return;
                const ctx = canvas.getContext('2d');

                try {
                    // event.data is the raw JPEG Blob from Python
                    // createImageBitmap is insanely fast because it decodes off the main thread
                    const bitmap = await createImageBitmap(event.data);
                    
                    // Paint the frame to the canvas
                    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                    
                    // Calculate and update FPS counter
                    frameCountRef.current += 1;
                    const now = performance.now();
                    if (now - lastTimeRef.current >= 1000) {
                        setFps(frameCountRef.current);
                        frameCountRef.current = 0;
                        lastTimeRef.current = now;
                    }
                } catch (err) {
                    console.error("Frame drop (decoding error):", err);
                }
            };

            ws.onclose = () => {
                console.log("🔴 Disconnected from Albedo Gateway");
                setIsConnected(false);
            };

            ws.onerror = (error) => {
                console.error("WebSocket Error:", error);
                setIsConnected(false);
            };
        } catch (securityErr) {
            console.warn("WebSocket initialization blocked (Sandbox environment), running in offline visual mode.", securityErr);
            setIsConnected(false);
        }

        // Cleanup on unmount
        return () => {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.close();
            }
        };
    }, []);

    const drawFallbackNoise = (x, y) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        
        // Draw a simulated background gradient
        const gradient = ctx.createRadialGradient(x * canvas.width, y * canvas.height, 10, x * canvas.width, y * canvas.height, 300);
        gradient.addColorStop(0, '#2d3748');
        gradient.addColorStop(1, '#1a202c');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add procedural noise to simulate the Taichi 1-SPP Engine
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 4) {
            const noise = (Math.random() - 0.5) * 50;
            data[i] = Math.max(0, Math.min(255, data[i] + noise));     
            data[i+1] = Math.max(0, Math.min(255, data[i+1] + noise)); 
            data[i+2] = Math.max(0, Math.min(255, data[i+2] + noise)); 
        }
        ctx.putImageData(imgData, 0, 0);
    };

    const handleMouseMove = (e) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        
        // Normalize mouse coordinates to a -1 to 1 range for the camera system
        const normalizedX = ((e.clientX - rect.left) / canvas.width) * 2 - 1;
        const normalizedY = -((e.clientY - rect.top) / canvas.height) * 2 + 1;

        if (!isConnected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            // If offline, draw simulated ray-tracing noise for the preview
            const rawX = (e.clientX - rect.left) / canvas.width;
            const rawY = (e.clientY - rect.top) / canvas.height;
            requestAnimationFrame(() => drawFallbackNoise(rawX, rawY));
            return;
        }
        
        // Send JSON data to the Node Gateway
        const payload = JSON.stringify({ x: normalizedX * 5, y: normalizedY * 5, z: 2.0 });
        wsRef.current.send(payload);
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-neutral-900 text-white font-sans overflow-hidden">
            
            {/* Header / Stats Overlay */}
            <div className="absolute top-4 left-4 flex flex-col gap-2 z-10">
                <h1 className="text-2xl font-bold tracking-tight text-white drop-shadow-md">
                    🪐 Project Albedo
                </h1>
                
                <div className="flex items-center gap-3 text-sm font-medium">
                    <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/40 backdrop-blur-sm border ${isConnected ? 'border-green-500/50 text-green-400' : 'border-red-500/50 text-red-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]' : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]'}`} />
                        {isConnected ? 'LIVE ENGINE' : 'OFFLINE'}
                    </span>
                    
                    <span className="px-3 py-1 rounded-full bg-black/40 backdrop-blur-sm border border-neutral-700 text-neutral-300">
                        {fps} FPS
                    </span>
                    
                    <span className="px-3 py-1 rounded-full bg-black/40 backdrop-blur-sm border border-neutral-700 text-neutral-300">
                        1-SPP Flow Matched
                    </span>
                </div>
            </div>

            {/* The Main Render Canvas */}
            <div className="relative rounded-lg overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] border border-neutral-800 transition-transform duration-300 hover:scale-[1.01]">
                <canvas 
                    ref={canvasRef}
                    width={512}
                    height={512}
                    onMouseMove={handleMouseMove}
                    className={`bg-black cursor-crosshair ${!isConnected ? 'opacity-50' : 'opacity-100'} transition-opacity duration-500`}
                />
                
                {/* Offline State Overlay */}
                {!isConnected && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                        <div className="flex flex-col items-center gap-3 animate-pulse">
                            <div className="w-8 h-8 border-4 border-t-blue-500 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" />
                            <p className="text-neutral-400 font-medium tracking-wide">Waiting for Taichi Engine...</p>
                        </div>
                    </div>
                )}
            </div>
            
            {/* Footer Instructions */}
            <p className="absolute bottom-6 text-neutral-500 text-sm font-medium tracking-wide">
                Move mouse over canvas to pan camera via WebSocket stream.
            </p>

        </div>
    );
}