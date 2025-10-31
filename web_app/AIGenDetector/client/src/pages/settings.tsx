import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useTheme } from "@/components/theme-provider";
import { useState } from "react";

export default function Settings() {
  const { theme, setTheme } = useTheme();
  const [modelPreference, setModelPreference] = useState("fine-tuned");

  return (
    <div className="space-y-8 max-w-3xl">
      <div>
        <h1 className="text-3xl font-bold text-foreground mb-2">Settings</h1>
        <p className="text-muted-foreground">
          Manage your preferences and application settings
        </p>
      </div>

      <Card className="p-6 space-y-6">
        <div>
          <h2 className="text-xl font-semibold mb-4 text-foreground">Appearance</h2>
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <Label htmlFor="dark-mode">Dark Mode</Label>
              <p className="text-sm text-muted-foreground">
                Toggle between light and dark theme
              </p>
            </div>
            <Switch
              id="dark-mode"
              checked={theme === "dark"}
              onCheckedChange={(checked) => setTheme(checked ? "dark" : "light")}
              data-testid="switch-dark-mode"
            />
          </div>
        </div>
      </Card>

      <Card className="p-6 space-y-6">
        <div>
          <h2 className="text-xl font-semibold mb-4 text-foreground">Model Settings</h2>
          <div className="space-y-4">
            <Label>Model Preference</Label>
            <RadioGroup
              value={modelPreference}
              onValueChange={setModelPreference}
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="fine-tuned" id="fine-tuned" data-testid="radio-fine-tuned" />
                <Label htmlFor="fine-tuned" className="font-normal cursor-pointer">
                  Fine-tuned Model (Recommended)
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="pretrained" id="pretrained" data-testid="radio-pretrained" />
                <Label htmlFor="pretrained" className="font-normal cursor-pointer">
                  Pre-trained Model
                </Label>
              </div>
            </RadioGroup>
          </div>
        </div>
      </Card>

      <Card className="p-6 space-y-6">
        <div>
          <h2 className="text-xl font-semibold mb-4 text-foreground">Account</h2>
          <div className="space-y-4">
            <div>
              <Label className="text-sm text-muted-foreground">Email</Label>
              <p className="text-base text-foreground">user@example.com</p>
            </div>
            <div>
              <Label className="text-sm text-muted-foreground">Member Since</Label>
              <p className="text-base text-foreground">January 2025</p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
