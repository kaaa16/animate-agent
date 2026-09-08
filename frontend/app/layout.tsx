import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Animate Agent Document Parser",
  description: "Parse technical documentation into DocumentIR",
  icons: { icon: "/icon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
