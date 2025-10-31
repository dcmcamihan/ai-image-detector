import { ThemeToggle } from "../theme-toggle";
import { ThemeProvider } from "../theme-provider";

export default function ThemeToggleExample() {
  return (
    <ThemeProvider>
      <div className="flex items-center justify-center h-32">
        <ThemeToggle />
      </div>
    </ThemeProvider>
  );
}
