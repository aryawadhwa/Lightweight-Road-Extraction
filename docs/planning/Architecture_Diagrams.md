# System Architecture Diagrams: GeoScribe

These diagrams formally visualize the data pipelines and the novel intersection matrix required for the SOTA Paper and Patent application.

## 1. High-Level Spatial Computing Engine

This diagram illustrates the separation between the static AI generation and the real-time dynamic data ingestion, converging at the fusion matrix.

```mermaid
graph TD
    subgraph "Static AI Core (Automated via AI-Researcher)"
        A[High-Res Satellite Imagery] -->|SegFormer| B(Pixel Probability Mask)
        B -->|Sat2Graph GNN| C{Mathematical Topology Graph}
        C -->|GeoJSON/Vector| D[(Graph Database)]
    end

    subgraph "Dynamic Ingestion Middleware (Node.js/Go)"
        E[Space-Track API] -->|TLE sets| I[Normalizer & Bounding Box Filter]
        F[OpenSky API] -->|Raw Transponder Data| I
        G[USGS API] -->|Seismic Polygons| I
        H[Austin CCTV] -->|Live Feed Metadata| I
        I -->|Sanitized Feature Stream| J((WebSocket Broadcaster))
    end

    subgraph "The Novel Fusion Matrix (Client WebGL)"
        D -->|Base Map Layer| K[Interaction Engine]
        J -->|Real-Time Overlays| K
        K -->|Spatial Collision Detection| L((Affected Edge Calculator))
        L --> M[Automated Rerouting & Impact Assessment Interface]
    end

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#ff9999,stroke:#333,stroke-width:4px
```

## 2. Ingestion & Polling Strategy (Avoiding API Limits)

This sequence diagram details how the middleware protects the API rate limits while serving high-frequency data to the clients.

```mermaid
sequenceDiagram
    participant C as WebGL Client (Viewport)
    participant M as Ingestion Middleware (Cache)
    participant API as External APIs (OpenSky, USGS)

    Note over C,M: Client connects and sends spatial Bounding Box (BBox)
    C->>M: WebSocket Handshake + Viewport BBox

    loop Every 5 Seconds (Middleware Polling Loop)
        M->>API: Fetch updates for global dataset
        API-->>M: Heavy Raw Payload
        Note over M: Middleware filters payload against Client's BBox
        M->>M: Compute Intersections & Drop out-of-bounds data
        M-->>C: Stream lightweight, localized GeoJSON
    end

    Note over C: Client pans map (New BBox)
    C->>M: Update Viewport BBox
    M-->>C: Stream localized GeoJSON for new area
```

## 3. The "Inventive Step" Spatial Collision

This diagram focuses specifically on the patentable aspect: how an external event dynamically alters the autonomously generated road network.

```mermaid
flowchart LR
    Event((USGS Earthquake Ping)) -->|Extract Lat/Lon & Magnitude| ComputeRadius[Generate Impact Polygon]

    RoadGraph[(AI Road Topologies)] --> ExtractEdges[Load Surrounding Edges]

    ComputeRadius --> SpatialIntersect{Does Edge intersect Polygon?}
    ExtractEdges --> SpatialIntersect

    SpatialIntersect -->|Yes| Sever[Invalidate Edge / Infinite Weight]
    SpatialIntersect -->|No| Keep[Maintain Base Resistance]

    Sever --> Recalculate[Run A* / Dijkstra on modified Graph]
    Keep --> Recalculate

    Recalculate --> UI[Push Emergency Routing to Base Layer]
```
