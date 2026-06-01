module rollout_worker_fsm (
    input  logic                  clk,
    input  logic                  rst_n,
    input  desc_pkg::desc_t       in_desc,
    input  logic                  in_valid,
    input  logic                  out_ready,
    output logic                  in_ready,
    output desc_pkg::completion_t out_comp,
    output logic                  out_valid
);

  import desc_pkg::*;

  typedef enum logic [1:0] {
    S_IDLE,
    S_DECODE,
    S_COMPLETE
  } worker_state_t;

  worker_state_t state;

  desc_t       active_desc;
  completion_t comp_reg;
  logic [15:0] token_count;

  localparam logic [15:0] REWARD_BOUNDARY = 16'd32;

  assign in_ready = (state == S_IDLE);
  assign out_valid = (state == S_COMPLETE);
  assign out_comp = comp_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      active_desc <= '0;
      comp_reg <= '0;
      token_count <= '0;
    end else begin
      unique case (state)
        S_IDLE: begin
          if (in_valid && in_ready) begin
            active_desc <= in_desc;
            token_count <= in_desc.seq_len;

            if (in_desc.opcode == DESC_OP_STOP) begin
              comp_reg.rollout_id   <= in_desc.rollout_id;
              comp_reg.status       <= COMP_STATUS_STOPPED;
              comp_reg.final_seq_len <= in_desc.seq_len;
              comp_reg.reward_id    <= in_desc.reward_model_id;
              state <= S_COMPLETE;
            end else if (in_desc.opcode == DESC_OP_DECODE) begin
              state <= S_DECODE;
            end else begin
              comp_reg.rollout_id   <= in_desc.rollout_id;
              comp_reg.status       <= COMP_STATUS_DONE;
              comp_reg.final_seq_len <= in_desc.seq_len;
              comp_reg.reward_id    <= in_desc.reward_model_id;
              state <= S_COMPLETE;
            end
          end
        end

        S_DECODE: begin
          token_count <= token_count + 16'd1;

          if ((token_count + 16'd1) >= active_desc.max_tokens) begin
            comp_reg.rollout_id   <= active_desc.rollout_id;
            comp_reg.status       <= COMP_STATUS_DONE;
            comp_reg.final_seq_len <= token_count + 16'd1;
            comp_reg.reward_id    <= active_desc.reward_model_id;
            state <= S_COMPLETE;
          end else if (((token_count + 16'd1) % REWARD_BOUNDARY) == 16'd0) begin
            comp_reg.rollout_id   <= active_desc.rollout_id;
            comp_reg.status       <= COMP_STATUS_REWARD_NEEDED;
            comp_reg.final_seq_len <= token_count + 16'd1;
            comp_reg.reward_id    <= active_desc.reward_model_id;
            state <= S_COMPLETE;
          end
        end

        S_COMPLETE: begin
          if (out_valid && out_ready)
            state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
