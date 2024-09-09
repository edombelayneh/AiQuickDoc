/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:5000',
  },
  webpack: (config, { isServer }) => {
    // Ensure `onnxruntime-node` is not bundled in the client-side code
    if (!isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        "onnxruntime-node": false,
      };
    }

    return config;
  },

  async rewrites() {
    return [
      {
        source: '/api/python/:path*',
        destination: process.env.NODE_ENV === 'development'
          ? 'http://127.0.0.1:5000/api/python/:path*'  // Local Python server
          : '/api/python/:path*',  // Production: handled by Vercel
      },
    ];
  },
};

export default nextConfig;