import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Proxy API and WS to backend (matches Vite dev proxy)
  async rewrites() {
    const simulatorUrl = process.env.SIMULATOR_URL || "http://localhost:8082";
    return [
      { source: "/api/:path*", destination: `${simulatorUrl}/api/:path*` },
      { source: "/ws", destination: `${simulatorUrl}/ws` },
    ];
  },

  // Turbopack resolve alias for mapbox-gl (SSR compatibility)
  turbopack: {
    root: __dirname,
    resolveAlias: {
      "mapbox-gl": "mapbox-gl/dist/mapbox-gl.js",
    },
  },

  // Webpack alias for mapbox-gl (production builds)
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "mapbox-gl$": "mapbox-gl/dist/mapbox-gl.js",
    };
    config.resolve.modules = [
      path.resolve(__dirname, "node_modules"),
      "node_modules",
    ];
    // Prevent webpack from resolving modules outside the frontend directory.
    // Without this, the parent package.json causes tailwindcss to be resolved
    // from the wrong root (which has no node_modules).
    config.resolve.roots = [__dirname];
    return config;
  },

  // Allow map tile domains
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "api.mapbox.com",
        pathname: "/styles/**",
      },
      {
        protocol: "https",
        hostname: "unpkg.com",
      },
    ],
  },
};

export default nextConfig;
