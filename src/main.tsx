import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

window.addEventListener('unhandledrejection', (event) => {
  const reasonStr = String(event?.reason?.message || event?.reason || '');
  if (
    reasonStr.includes('WebSocket') ||
    reasonStr.includes('websocket') ||
    reasonStr.includes('ws') ||
    reasonStr.includes('closed without opened')
  ) {
    event.preventDefault();
    event.stopImmediatePropagation();
  }
});

window.addEventListener('error', (event) => {
  const msg = String(event?.message || event?.error || '');
  if (
    msg.includes('WebSocket') ||
    msg.includes('websocket') ||
    msg.includes('closed without opened')
  ) {
    event.preventDefault();
    event.stopImmediatePropagation();
  }
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
