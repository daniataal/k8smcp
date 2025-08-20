"use client"

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Image as ImageIcon, Sparkles, Loader2, XCircle } from 'lucide-react';

export function ModelInference() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setPrediction(null);
      setConfidence(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError("Please select an image first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setConfidence(null);

    try {
      const reader = new FileReader();
      reader.readAsDataURL(selectedImage);
      reader.onloadend = async () => {
        const base64Image = (reader.result as string).split(',')[1]; // Get base64 string without data:image part

        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: base64Image }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.prediction !== undefined) {
          setPrediction(data.prediction);
          setConfidence(data.confidence);
        } else {
          setError("Prediction data not found in response.");
        }
      };
    } catch (e: any) {
      console.error("Prediction failed:", e);
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="glass-card border-primary/10">
      <CardHeader className="pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <div>
            <CardTitle className="text-xl">Model Inference</CardTitle>
            <p className="text-muted-foreground text-sm">Upload an image to get an ML prediction</p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid w-full items-center gap-1.5">
          <Label htmlFor="picture">Upload Image</Label>
          <Input id="picture" type="file" accept="image/*" onChange={handleImageChange} />
        </div>

        {imagePreview && (
          <div className="flex flex-col items-center space-y-4">
            <h3 className="text-lg font-semibold">Image Preview:</h3>
            <img src={imagePreview} alt="Image Preview" className="max-w-xs max-h-48 rounded-lg shadow-md" />
            <Button onClick={handlePredict} disabled={isLoading || !selectedImage}>
              {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="mr-2 h-4 w-4" />
              )}
              {isLoading ? "Predicting..." : "Get Prediction"}
            </Button>
          </div>
        )}

        {(prediction !== null || error) && ( // Show prediction or error if available
          <>
            <Separator />
            <div className="text-center space-y-2">
              {error ? (
                <p className="text-red-500 flex items-center justify-center gap-2">
                  <XCircle className="w-5 h-5" />
                  Error: {error}
                </p>
              ) : (
                <>
                  <p className="text-2xl font-bold">Predicted Digit: {prediction}</p>
                  <p className="text-muted-foreground">Confidence: {(confidence! * 100).toFixed(2)}%</p>
                </>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
