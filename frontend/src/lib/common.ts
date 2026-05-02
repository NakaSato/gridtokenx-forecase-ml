import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { Reading, ApiError } from "./types";

/**
 * Merge Tailwind CSS classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Calculate total energy in MW from readings
 */
export function calculateEnergyMW(
  readings: Reading[],
  field: "energy_generated" | "energy_consumed"
): number {
  return readings.reduce((sum, r) => sum + (r[field] || 0), 0) / 1000;
}

/**
 * Create API error object
 */
export function createApiError(message: string, code?: string): ApiError {
  return { 
    message, 
    code,
    timestamp: new Date().toISOString()
  };
}

/**
 * Format timestamp to readable string
 */
export function formatTimestamp(timestamp?: string | Date): string {
  const date = timestamp 
    ? (typeof timestamp === "string" ? new Date(timestamp) : timestamp)
    : new Date();
  return date.toLocaleTimeString("en-US", { 
    hour12: false, 
    hour: "2-digit", 
    minute: "2-digit", 
    second: "2-digit" 
  });
}
