import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The data bundle is static: viz/export.py writes it to public/data/, which Vite serves at /data/.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
