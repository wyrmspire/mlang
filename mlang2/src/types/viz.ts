export interface VizWindow {
  x_price_1m: number[][]; // [open, high, low, close, volume]
  x_price_5m: number[][];
  x_price_15m: number[][];
  x_context: number[];
  norm_method: string;
  norm_params: Record<string, any>;
}

export interface VizOCO {
  entry_price: number;
  stop_price: number;
  tp_price: number;
  entry_type: string;
  direction: string;
  reference_type: string;
  reference_value: number;
  atr_at_creation: number;
  max_bars: number;
  stop_atr: number;
  tp_multiple: number;
}

export interface VizDecision {
  decision_id: string;
  timestamp: string | null;
  bar_idx: number;
  index: number;
  scanner_id: string;
  scanner_context: Record<string, any>;
  action: string;
  skip_reason: string;
  current_price: number;
  atr: number;
  cf_outcome: string;
  cf_pnl_dollars: number;
  window?: VizWindow | null;
  oco?: VizOCO | null;
}

export interface VizFill {
  order_id: string;
  fill_type: string;
  price: number;
  bar_idx: number;
  timestamp?: string | null;
}

export interface VizTrade {
  trade_id: string;
  decision_id: string;
  index: number;
  direction: string;
  size: number;
  entry_time: string | null;
  entry_bar: number;
  entry_price: number;
  exit_time: string | null;
  exit_bar: number;
  exit_price: number;
  exit_reason: string;
  outcome: string;
  pnl_points: number;
  pnl_dollars: number;
  r_multiple: number;
  bars_held: number;
  mae: number;
  mfe: number;
  fills: VizFill[];
}

export interface RunManifest {
  run_id: string;
  start_time: string;
  config: any;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export type UIActionType = 'SET_INDEX' | 'SET_FILTER' | 'SET_MODE' | 'LOAD_RUN' | 'RUN_STRATEGY';

export interface UIAction {
  type: UIActionType;
  payload: any;
}

export interface AgentResponse {
  reply: string;
  ui_action?: UIAction;
}