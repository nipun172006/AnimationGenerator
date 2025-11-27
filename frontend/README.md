# Frontend (React + TypeScript)

Simple UI to send JSON animation instructions to the FastAPI backend.

## Setup

```bash
cd frontend
npm install
npm run dev
```

- Dev server runs at `http://localhost:5173` (default Vite).
- Backend FastAPI should run at `http://localhost:8000`.

If you need a dev proxy (CORS avoidance), create `vite.config.ts`:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/render': 'http://localhost:8000',
      '/health': 'http://localhost:8000'
    }
  }
})
```

Then update `package.json` devDependencies to include `@vitejs/plugin-react` and install it:

```bash
npm i -D @vitejs/plugin-react
```

Or keep direct fetch to `http://localhost:8000` and allow CORS on backend if needed.
