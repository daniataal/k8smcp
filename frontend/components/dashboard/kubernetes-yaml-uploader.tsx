"use client"

import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { FileUp, Loader2, CheckCircle, XCircle } from 'lucide-react';

export function KubernetesYamlUploader() {
  const [yamlContent, setYamlContent] = useState<string>('');
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        setYamlContent(e.target?.result as string);
        setUploadStatus('idle');
        setMessage(null);
      };
      reader.readAsText(file);
    } else {
      setYamlContent('');
      setFileName(null);
    }
  };

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
  }, []);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        setYamlContent(e.target?.result as string);
        setUploadStatus('idle');
        setMessage(null);
      };
      reader.readAsText(file);
    }
  }, []);

  const handleUpload = async () => {
    if (!yamlContent.trim()) {
      setMessage("Please provide YAML content or upload a file.");
      setUploadStatus('error');
      return;
    }

    setUploadStatus('loading');
    setMessage(null);

    try {
      const response = await fetch('/api/apply-yaml', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ yaml: yamlContent }),
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus('success');
        setMessage(data.message || "YAML applied successfully!");
        setYamlContent(''); // Clear content on success
        setFileName(null);
      } else {
        setUploadStatus('error');
        setMessage(data.error || "Failed to apply YAML.");
      }
    } catch (e: any) {
      console.error("Error applying YAML:", e);
      setUploadStatus('error');
      setMessage(`Network error or server unreachable: ${e.message}`);
    }
  };

  return (
    <Card className="glass-card border-primary/10">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
            <FileUp className="w-4 h-4 text-primary" />
          </div>
          <div>
            <CardTitle className="text-xl">Deploy Kubernetes YAML</CardTitle>
            <CardDescription>Upload a YAML file to apply Kubernetes resources</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div
          className="border-2 border-dashed border-muted-foreground/30 rounded-lg p-6 text-center cursor-pointer hover:border-primary transition-colors"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById('yaml-file-upload')?.click()}
        >
          <Input
            id="yaml-file-upload"
            type="file"
            accept=".yaml,.yml"
            onChange={handleFileChange}
            className="hidden"
          />
          <FileUp className="mx-auto h-12 w-12 text-muted-foreground" />
          <p className="text-muted-foreground mt-2">Drag & drop your YAML file here, or click to browse</p>
          {fileName && <p className="text-sm text-primary mt-1">Selected file: {fileName}</p>}
        </div>

        <div className="grid w-full gap-1.5">
          <Label htmlFor="yaml-content">Or paste YAML content directly</Label>
          <Textarea
            id="yaml-content"
            placeholder="apiVersion: v1\nkind: Pod\nmetadata:\n  name: my-pod\nspec:\n  containers:\n    - name: my-container\n      image: busybox"
            value={yamlContent}
            onChange={(e) => setYamlContent(e.target.value)}
            rows={10}
            className="font-mono"
          />
        </div>

        <Button onClick={handleUpload} disabled={uploadStatus === 'loading' || !yamlContent.trim()}>
          {uploadStatus === 'loading' ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <FileUp className="mr-2 h-4 w-4" />
          )}
          {uploadStatus === 'loading' ? "Applying YAML..." : "Apply YAML to Kubernetes"}
        </Button>

        {message && (
          <div
            className={`p-3 rounded-md flex items-center gap-2 ${
              uploadStatus === 'success' ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
            }`}
          >
            {uploadStatus === 'success' ? (
              <CheckCircle className="h-5 w-5" />
            ) : (
              <XCircle className="h-5 w-5" />
            )}
            <span>{message}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
