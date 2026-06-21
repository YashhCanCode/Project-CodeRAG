import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CodeRAG",
  description: "Ask questions about your codebases and technical docs, with citations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
