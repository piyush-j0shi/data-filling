import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // This allows the ngrok URL to communicate with your local Vite server
    allowedHosts: [
      '.ngrok-free.app', // Permits all ngrok subdomains
      '.ngrok-free.dev'  // Permits ngrok's alternative dev TLD
    ]
  }
})