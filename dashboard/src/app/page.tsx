"use client";

import { useState, useEffect } from "react";
import { Video, Bell, LayoutDashboard, Activity, CheckCircle2 } from "lucide-react";

interface TrackEvent {
  global_id: number;
  camera_id: string;
  track_id: number;
  class_label: string;
  color: string;
  type?: string;
  timestamp: number;
}

function StreamImage({ src, alt, className }: { src: string, alt: string, className: string }) {
  const [retry, setRetry] = useState(0);

  return (
    <img 
      src={`${src}?retry=${retry}`} 
      alt={alt}
      className={className}
      onError={() => {
        // If the backend stream is not ready, retry every 2 seconds instead of permanently hiding
        setTimeout(() => setRetry(r => r + 1), 2000);
      }}
    />
  );
}

export default function Dashboard() {
  const [events, setEvents] = useState<TrackEvent[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const sse = new EventSource("http://localhost:8000/events");
    
    sse.onopen = () => setConnected(true);
    sse.onerror = () => setConnected(false);
    
    sse.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setEvents((prev) => [data, ...prev].slice(0, 100));
      } catch (err) {
        console.error(err);
      }
    };

    return () => sse.close();
  }, []);

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-300 font-sans">
      {/* Top Navbar */}
      <header className="h-16 border-b border-zinc-800 bg-zinc-900/50 flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/20 text-primary rounded-lg">
            <LayoutDashboard size={20} />
          </div>
          <h1 className="text-xl font-semibold text-zinc-100 tracking-tight">CCTV Command Center</h1>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-800/50 border border-zinc-700/50">
            {connected ? (
              <><CheckCircle2 size={14} className="text-accent" /><span className="text-zinc-400">System Online</span></>
            ) : (
              <><Activity size={14} className="text-red-500" /><span className="text-zinc-400">Connecting...</span></>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Left: Camera Grid */}
        <main className="flex-1 p-6 overflow-y-auto">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-medium text-zinc-100">Live Feeds</h2>
            <div className="text-xs text-zinc-500 font-mono">2 ACTIVE NODES</div>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {["cam_1", "cam_2"].map((cam, idx) => (
              <div key={cam} className="rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden shadow-sm flex flex-col">
                <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 bg-zinc-900/80">
                  <div className="flex items-center gap-2">
                    <Video size={16} className="text-zinc-500" />
                    <span className="font-mono text-xs font-semibold text-zinc-300 uppercase">{cam}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                    </span>
                    <span className="text-[10px] uppercase font-bold text-red-500 tracking-wider">Live</span>
                  </div>
                </div>
                <div className="relative aspect-video bg-black flex items-center justify-center">
                  <Video size={32} className="opacity-20 absolute z-0" />
                  <StreamImage 
                    src={`http://localhost:${8001 + idx}/mjpeg`} 
                    alt={`${cam} live feed`}
                    className="w-full h-full object-contain relative z-10"
                  />
                </div>
              </div>
            ))}
          </div>
        </main>

        {/* Right: Event Log Sidebar */}
        <aside className="w-96 border-l border-zinc-800 bg-zinc-900/30 flex flex-col shrink-0">
          <div className="p-4 border-b border-zinc-800 bg-zinc-900/50 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2">
              <Bell size={16} className="text-zinc-400" />
              <h2 className="text-sm font-medium text-zinc-200">Global Registry Log</h2>
            </div>
            <span className="px-2 py-0.5 rounded-md bg-zinc-800 text-[10px] font-mono text-zinc-400 border border-zinc-700">
              {events.length} EVENTS
            </span>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-hide">
            {events.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-zinc-600">
                <Activity size={32} className="mb-3 opacity-20" />
                <p className="text-sm font-medium">Awaiting detections...</p>
              </div>
            ) : (
              events.map((evt, i) => (
                <div 
                  key={`${evt.global_id}-${evt.timestamp}-${i}`}
                  className="bg-zinc-900 border border-zinc-800/80 p-3 rounded-lg hover:border-zinc-700 transition-colors flex items-start gap-3"
                >
                  <div className="w-10 h-10 rounded-md bg-zinc-800 border border-zinc-700 flex items-center justify-center shrink-0">
                    <span className="font-mono font-bold text-zinc-300">#{evt.global_id}</span>
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm font-medium text-zinc-200 truncate capitalize">
                        {evt.color} {evt.type || evt.class_label}
                      </p>
                      <span className="text-[10px] font-mono text-zinc-500 whitespace-nowrap ml-2">
                        {new Date(evt.timestamp * 1000).toLocaleTimeString([], { hour12: false })}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-zinc-500">
                      <span className="px-1.5 py-0.5 rounded bg-zinc-800/80 font-mono uppercase border border-zinc-700/50">
                        {evt.camera_id}
                      </span>
                      <span>Track: {evt.track_id}</span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
