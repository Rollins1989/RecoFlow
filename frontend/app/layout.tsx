import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "RecoFlow",
  description: "Real-time personalized recommendation & ranking platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
