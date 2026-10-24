# Web console (Next.js)

The underwriter-facing console for the Credit Risk Intelligence Platform. It
is a pure client of the FastAPI `/api/v1` contract: it never imports model
code, and everything it shows comes from the API.

| Route | What it does | API |
|---|---|---|
| `/` | Executive dashboard: volume, approval rate, risk mix, champion AUC/Gini/KS, latency, API health | `/monitoring/*`, `/models/versions`, `/predict/history`, `/health` |
| `/predict` | Score an application: calibrated PD, scorecard points, decision and, for declines, ECOA/Reg B adverse-action reasons | `POST /predict` |
| `/validation` | Model validation: out-of-time AUC/Gini/KS with bootstrap CIs, gains chart, decision threshold, fair-lending (four-fifths) review, candidate models | `GET /models/validation-report` |
| `/login`, `/register` | JWT auth; the profile and role come from `GET /auth/me` | `/auth/*` |

## Run it

```bash
cp .env.example .env.local   # NEXT_PUBLIC_API_BASE_URL, default http://localhost:8000
npm ci
npm run dev                  # http://localhost:3000
```

The API must be running with a promoted champion (`make train` at the repo
root). With Docker, `docker compose up` starts it as the `web` service on
port 3000.

## Checks

```bash
npm run lint
npx tsc --noEmit
npm run build
```

CI runs all three on every push and pull request.

## Structure

```
src/
  app/(auth)/          login, register
  app/(dashboard)/     page.tsx (dashboard), predict/, validation/
  components/ui/       Radix + Tailwind + CVA primitives (shadcn-style)
  components/layout/   sidebar, top bar, command palette (Ctrl/Cmd+K), user menu
  components/dashboard KPI cards and charts
  hooks/               React Query hooks: use-auth, use-dashboard, use-credit
  lib/types.ts         TypeScript mirror of api/schemas.py
  config/nav.ts        navigation
```

Stack: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4,
TanStack Query, Zustand, Recharts, Radix UI, Framer Motion.
