import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8200',
      '/kis': 'http://127.0.0.1:8200',
      '/exact-frame': 'http://127.0.0.1:8200',
      '/videos': 'http://127.0.0.1:8200',
      '/vqa': 'http://127.0.0.1:8200',
      '/trake': 'http://127.0.0.1:8200',
      '/feedback': 'http://127.0.0.1:8200',
      '/frames': 'http://127.0.0.1:8000',
      '/keyframe-image': 'http://127.0.0.1:8000',
      '/media': 'http://127.0.0.1:8000',
    },
  },
})
