#!/usr/bin/env node
// Generates a self-signed TLS certificate for local/LAN development so the
// browser mic (getUserMedia) works from devices other than localhost —
// Chrome and friends only allow mic access in a "secure context", and any
// origin besides localhost/127.0.0.1 needs HTTPS to qualify.
//
// This is a dev convenience, not a production TLS setup: browsers will warn
// on first visit until you accept/trust the certificate on each device. For
// a public deployment, put a real certificate (e.g. via Let's Encrypt/Caddy)
// in front of the app instead.
//
//   npm run cert

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getLanIPv4s } from '../src/net.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const certDir = path.join(__dirname, '..', 'certs');
const keyPath = path.join(certDir, 'dev-key.pem');
const certPath = path.join(certDir, 'dev-cert.pem');

mkdirSync(certDir, { recursive: true });

const sans = [
  'DNS:localhost',
  'IP:127.0.0.1',
  ...getLanIPv4s().map((ip) => `IP:${ip}`),
];

const result = spawnSync(
  'openssl',
  [
    'req', '-x509', '-newkey', 'rsa:2048', '-sha256', '-days', '825', '-nodes',
    '-keyout', keyPath,
    '-out', certPath,
    '-subj', '/CN=whisper-flow-dev',
    '-addext', `subjectAltName=${sans.join(',')}`,
  ],
  { stdio: 'inherit' },
);

if (result.error || result.status !== 0) {
  console.error(
    '\nCould not generate a certificate — is `openssl` installed and on PATH?',
  );
  process.exit(1);
}

console.log(`\nWrote ${certPath} and ${keyPath} (covers: ${sans.join(', ')}).`);
console.log(
  'Run `npm start` — the server will pick these up automatically and switch to HTTPS.',
);
console.log(
  "Browsers will warn on first visit (self-signed cert); accept the risk once per device to continue.",
);
