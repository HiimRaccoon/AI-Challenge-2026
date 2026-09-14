/**
 * Client-side Telemetry Recorder & Storage
 * Records operator actions, latencies, task modes, and query lifecycles.
 */

export interface ClientTelemetryRecord {
  timestamp: string
  action: string
  taskMode: string
  queryId?: string | null
  queryText?: string | null
  videoId?: string | null
  frameId?: number | null
  latencyMs?: number
  details?: Record<string, unknown>
}

class TelemetryManager {
  private records: ClientTelemetryRecord[] = []
  private maxRecords = 1000

  public record(event: Omit<ClientTelemetryRecord, 'timestamp'>): void {
    const entry: ClientTelemetryRecord = {
      ...event,
      timestamp: new Date().toISOString(),
    }
    this.records.push(entry)
    if (this.records.length > this.maxRecords) {
      this.records.shift()
    }
    // Also log to window for inspection
    if (typeof window !== 'undefined') {
      ;(window as unknown as { __TELEMETRY_LOGS__?: ClientTelemetryRecord[] }).__TELEMETRY_LOGS__ = this.records
    }
  }

  public getRecords(): ClientTelemetryRecord[] {
    return [...this.records]
  }

  public clear(): void {
    this.records = []
  }

  public exportJsonl(): string {
    return this.records.map((r) => JSON.stringify(r)).join('\n')
  }
}

export const telemetry = new TelemetryManager()
