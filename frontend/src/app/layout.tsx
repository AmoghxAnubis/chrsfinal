"use client";

import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/auth-context";
import Head from "next/head";
import { Toaster } from "sonner";

const inter = Inter({ 
  subsets: ["latin"],
  variable: '--font-inter',
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable}`}>
      <Head>
        <title>CHRS - Comprehensive Health Research System</title>
        <meta name="description" content="AI-Powered Drug Discovery Platform" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <body className={`${inter.className} bg-[#0f0c29] text-white min-h-screen`}>
        <AuthProvider>
          {children}
          <Toaster 
            position="top-center"
            richColors
            theme="dark"
            toastOptions={{
              classNames: {
                toast: 'bg-[#1e1b4b] border border-purple-500/30',
                title: 'text-purple-100',
                description: 'text-purple-200',
                actionButton: 'bg-purple-600 hover:bg-purple-700',
                cancelButton: 'bg-[#3a1c6e] hover:bg-[#3a1c6e]/80',
              }
            }}
          />
        </AuthProvider>
      </body>
    </html>
  );
}