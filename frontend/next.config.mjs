/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*', // Match any request to /api/ and beyond
        destination: 'http://localhost:8080/:path*', // Proxy to your Flask backend
      },
    ];
  },
}

export default nextConfig
