module desc_ring #(
    parameter int DEPTH = 16,
    parameter int PTR_W = 4
) (
    input  logic            clk,
    input  logic            rst_n,
    input  desc_pkg::desc_t push_desc,
    input  logic            push_valid,
    input  logic            pop_ready,
    output logic            push_ready,
    output desc_pkg::desc_t pop_desc,
    output logic            pop_valid,
    output logic            full,
    output logic            empty
);

  import desc_pkg::*;

  desc_t mem [0:DEPTH-1];
  logic [PTR_W:0] head;
  logic [PTR_W:0] tail;

  logic push_fire;
  logic pop_fire;

  assign empty = (head == tail);
  assign full  = (head[PTR_W-1:0] == tail[PTR_W-1:0]) &&
                 (head[PTR_W] != tail[PTR_W]);

  assign push_ready = !full;
  assign pop_valid  = !empty;
  assign pop_desc   = mem[head[PTR_W-1:0]];

  assign push_fire = push_valid && push_ready;
  assign pop_fire  = pop_valid && pop_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      head <= '0;
      tail <= '0;
    end else begin
      if (push_fire) begin
        mem[tail[PTR_W-1:0]] <= push_desc;
        tail <= tail + 1'b1;
      end
      if (pop_fire)
        head <= head + 1'b1;
    end
  end

endmodule
