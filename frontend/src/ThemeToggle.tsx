import { useEffect, useState } from "react";

// Light/dark switch. Until the user picks one, the page follows the system setting; a pick is saved in
// localStorage (index.html re-applies it before the first paint) and set as data-theme on <html>.
type Theme = "light" | "dark";
const KEY = "orbitbot-theme";
const systemDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;

function saved(): Theme | null {
  try {
    const t = localStorage.getItem(KEY);
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null; // storage blocked (private mode, sandboxed preview): just don't remember
  }
}

export default function ThemeToggle() {
  const [choice, setChoice] = useState<Theme | null>(saved);
  const [system, setSystem] = useState<Theme>(systemDark() ? "dark" : "light");
  const theme = choice ?? system;

  // follow system changes while the user hasn't picked a theme
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setSystem(mq.matches ? "dark" : "light");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (choice) document.documentElement.dataset.theme = choice;
    else delete document.documentElement.dataset.theme;
  }, [choice]);

  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setChoice(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* not remembered, still applied */
    }
  };

  const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  return (
    <button className="btn ghost theme-btn" onClick={toggle} aria-label={label} title={label}>
      {theme === "dark" ? (
        // sun: shown in dark mode, switches to light
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      ) : (
        // moon: shown in light mode, switches to dark
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        </svg>
      )}
    </button>
  );
}
