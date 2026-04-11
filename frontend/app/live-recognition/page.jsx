"use client";

import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import "./page.css";

export default function Page() {
  const webcamRef = useRef(null);
  const wrapperRef = useRef(null);
  const canvasRef = useRef(null);
  const requestInFlight = useRef(false);

  const [faces, setFaces] = useState([]);
  const [frameSize, setFrameSize] = useState({ width: 1, height: 1 });
  const [displaySize, setDisplaySize] = useState({ width: 1, height: 1 });
  const [mainIdentity, setMainIdentity] = useState("Analyzing...");
  const [mainStatus, setMainStatus] = useState("SCANNING");
  const [mainConfidence, setMainConfidence] = useState("--");

  useEffect(() => {
    const updateDisplaySize = () => {
      if (!wrapperRef.current) return;
      const rect = wrapperRef.current.getBoundingClientRect();
      setDisplaySize({
        width: rect.width,
        height: rect.height,
      });
    };

    updateDisplaySize();
    window.addEventListener("resize", updateDisplaySize);
    return () => window.removeEventListener("resize", updateDisplaySize);
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (requestInFlight.current) return;
      if (!webcamRef.current) return;

      const imageSrc = webcamRef.current.getScreenshot({
        width: 640,
        height: 360,
      });

      if (!imageSrc) return;

      requestInFlight.current = true;

      try {
        const res = await fetch("http://127.0.0.1:8000/api/recognize-frame", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ image: imageSrc }),
        });

        const data = await res.json();
        const detectedFaces = data.faces || [];

        setFaces(detectedFaces);
        setFrameSize({
          width: data.frame_width || 1,
          height: data.frame_height || 1,
        });

        if (detectedFaces.length === 0) {
          setMainIdentity("No Face Detected");
          setMainStatus("SCANNING");
          setMainConfidence("--");
        } else {
          const primaryFace = detectedFaces[0];
          const safeName =
            primaryFace.name && primaryFace.name.toLowerCase() !== "unknown"
              ? primaryFace.name
              : "Unknown";

          setMainIdentity(safeName);

          if (safeName === "Unknown") {
            setMainStatus("UNKNOWN USER");
          } else {
            setMainStatus("ACCESS GRANTED");
          }

          if (typeof primaryFace.confidence === "number") {
            setMainConfidence(`${(primaryFace.confidence * 100).toFixed(1)}%`);
          } else {
            setMainConfidence("--");
          }
        }
      } catch (err) {
        console.error("Recognition request failed:", err);
        setMainIdentity("Connection Error");
        setMainStatus("OFFLINE");
        setMainConfidence("--");
      } finally {
        requestInFlight.current = false;
      }
    }, 150);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = displaySize.width;
    canvas.height = displaySize.height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = displaySize.width / frameSize.width;
    const scaleY = displaySize.height / frameSize.height;

    for (const face of faces) {
      const x = face.box.x * scaleX;
      const y = face.box.y * scaleY;
      const w = face.box.w * scaleX;
      const h = face.box.h * scaleY;

      const safeName =
        face.name && face.name.toLowerCase() !== "unknown"
          ? face.name
          : "Unknown";

      const recognized = safeName !== "Unknown";
      const color = recognized ? "#22c55e" : "#ff4d6d";

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 16);
      ctx.stroke();

      const label = safeName;
      ctx.font = "600 14px Inter, sans-serif";
      const textWidth = ctx.measureText(label).width;
      const labelW = textWidth + 24;
      const labelH = 34;
      const labelX = x;
      const labelY = Math.max(10, y - 42);

      ctx.fillStyle = "rgba(10, 14, 20, 0.88)";
      ctx.beginPath();
      ctx.roundRect(labelX, labelY, labelW, labelH, 12);
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, labelX + 12, labelY + 22);
    }
  }, [faces, frameSize, displaySize]);

  const statusClass =
    mainStatus === "ACCESS GRANTED"
      ? "success"
      : mainStatus === "UNKNOWN USER"
      ? "danger"
      : "scanning";

  const identityClass = mainIdentity === "Unknown" ? "danger-text" : "success-text";

  return (
    <div className="page">
      <div className="app-shell">
        <section className="camera-panel">
          <div className="camera-header">
            <div>
              <p className="eyebrow">Biometric Security Interface</p>
              <h1 className="title">FaceGate Live Recognition</h1>
              <p className="subtitle">
                Real-time facial verification and identity screening
              </p>
            </div>

            <div className={`status-pill ${statusClass}`}>{mainStatus}</div>
          </div>

          <div ref={wrapperRef} className="camera-wrapper">
            <Webcam
              ref={webcamRef}
              audio={false}
              mirrored={false}
              screenshotFormat="image/jpeg"
              forceScreenshotSourceSize={false}
              videoConstraints={{
                width: 1280,
                height: 720,
                facingMode: "user",
              }}
              className="webcam"
            />

            <div className="scan-frame" />
            <div className="scan-line" />

            <canvas ref={canvasRef} className="overlay-canvas" />
          </div>
        </section>

        <aside className="info-panel">
          <div className="identity-card">
            <p className="card-label">Identity Status</p>
            <h2 className={`identity-name ${identityClass}`}>{mainIdentity}</h2>
            <p className="identity-meta">
              {mainIdentity === "No Face Detected"
                ? "Waiting for a face inside the frame."
                : mainIdentity === "Unknown"
                ? "Face detected but no verified match was accepted."
                : mainIdentity === "Analyzing..."
                ? "System is processing live facial data."
                : "Verified user detected with stable recognition output."}
            </p>

            <div className="metrics">
              <div className="metric-card">
                <p className="metric-label">Confidence</p>
                <p className="metric-value">{mainConfidence}</p>
              </div>

              <div className="metric-card">
                <p className="metric-label">Faces Detected</p>
                <p className="metric-value">{faces.length}</p>
              </div>
            </div>
          </div>

          <div className="system-card">
            <p className="card-label">System Notes</p>
            <ul className="system-list">
              <li>Live camera feed active</li>
              <li>Recognition overlay enabled</li>
              <li>Biometric verification running</li>
            </ul>
          </div>
        </aside>
      </div>
    </div>
  );
}