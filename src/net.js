// Small shared helper: the machine's LAN-facing IPv4 addresses, used both to
// generate a dev certificate with the right SANs and to print URLs other
// devices on the network can actually reach.

import os from 'node:os';

export function getLanIPv4s() {
  const out = [];
  for (const addrs of Object.values(os.networkInterfaces())) {
    for (const addr of addrs || []) {
      if (addr.family === 'IPv4' && !addr.internal) out.push(addr.address);
    }
  }
  return out;
}
