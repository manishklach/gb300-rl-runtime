module rl_runtime_top #(
    parameter int RING_DEPTH = 16,
    parameter int RING_PTR_W = 4
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  desc_pkg::desc_t       host_desc,
    input  logic                  host_desc_valid,
    input  logic                  host_comp_ready,
    input  logic                  doorbell_pulse,
    output logic                  host_desc_ready,
    output desc_pkg::completion_t host_comp,
    output logic                  host_comp_valid
);

  import desc_pkg::*;

  desc_t       worker_desc;
  logic        worker_desc_valid;
  logic        worker_desc_ready;
  completion_t worker_comp;
  logic        worker_comp_valid;
  logic        worker_comp_ready;

  logic cmd_full;
  logic cmd_empty;
  logic comp_full;
  logic comp_empty;

  logic cmd_pop_ready;

  /*
   * v1 model note:
   * host_desc_valid directly feeds the descriptor ring push path.
   * doorbell_pulse is included so the top-level interface mirrors the
   * software fast path, but it is not yet required for ring visibility.
   */
  logic doorbell_seen;

  assign cmd_pop_ready = worker_desc_ready && worker_desc_valid;
  assign worker_comp_ready = !comp_full;
  assign doorbell_seen = doorbell_pulse;

  desc_ring #(
      .DEPTH(RING_DEPTH),
      .PTR_W(RING_PTR_W)
  ) u_desc_ring (
      .clk(clk),
      .rst_n(rst_n),
      .push_desc(host_desc),
      .push_valid(host_desc_valid),
      .pop_ready(cmd_pop_ready),
      .push_ready(host_desc_ready),
      .pop_desc(worker_desc),
      .pop_valid(worker_desc_valid),
      .full(cmd_full),
      .empty(cmd_empty)
  );

  rollout_worker_fsm u_worker (
      .clk(clk),
      .rst_n(rst_n),
      .in_desc(worker_desc),
      .in_valid(worker_desc_valid),
      .out_ready(worker_comp_ready),
      .in_ready(worker_desc_ready),
      .out_comp(worker_comp),
      .out_valid(worker_comp_valid)
  );

  completion_ring #(
      .DEPTH(RING_DEPTH),
      .PTR_W(RING_PTR_W)
  ) u_completion_ring (
      .clk(clk),
      .rst_n(rst_n),
      .push_comp(worker_comp),
      .push_valid(worker_comp_valid),
      .pop_ready(host_comp_ready),
      .push_ready(),
      .pop_comp(host_comp),
      .pop_valid(host_comp_valid),
      .full(comp_full),
      .empty(comp_empty)
  );

endmodule
