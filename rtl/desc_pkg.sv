package desc_pkg;

  localparam logic [7:0] DESC_OP_NOP    = 8'd0;
  localparam logic [7:0] DESC_OP_DECODE = 8'd1;
  localparam logic [7:0] DESC_OP_REWARD = 8'd2;
  localparam logic [7:0] DESC_OP_STOP   = 8'd255;

  typedef struct packed {
    logic [7:0]  opcode;
    logic [7:0]  flags;
    logic [15:0] rollout_id;
    logic [15:0] kv_arena_id;
    logic [15:0] prefix_id;
    logic [31:0] kv_offset;
    logic [31:0] delta_offset;
    logic [15:0] seq_len;
    logic [15:0] max_tokens;
    logic [15:0] reward_model_id;
    logic [15:0] reserved;
  } desc_t;

  typedef struct packed {
    logic [15:0] rollout_id;
    logic [7:0]  status;
    logic [15:0] final_seq_len;
    logic [15:0] reward_id;
  } completion_t;

  localparam logic [7:0] COMP_STATUS_DONE          = 8'h01;
  localparam logic [7:0] COMP_STATUS_REWARD_NEEDED = 8'h02;
  localparam logic [7:0] COMP_STATUS_STOPPED       = 8'hff;

endpackage
