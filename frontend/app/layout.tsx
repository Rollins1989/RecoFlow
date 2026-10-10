import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "RecoFlow",
  description: "A recommendation-systems demo with hybrid retrieval and offline ranking experiments",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
