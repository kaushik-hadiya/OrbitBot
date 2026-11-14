import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the API runs separately (uv run uvicorn api:app --port 7860) and Vite proxies to it,
// so the browser sees one origin, as it does in production where FastAPI serves the built files.
const API = "http://localhost:7860";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/chat": API, "/meta": API, "/kb": API, "/health": API },
  },
});
