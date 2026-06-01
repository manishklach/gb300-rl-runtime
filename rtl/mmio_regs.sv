module mmio_regs (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        wr_en,
    input  logic [7:0]  wr_addr,
    input  logic [31:0] wr_data,
    output logic        doorbell_pulse,
    output logic [31:0] doorbell_value
);

  localparam logic [7:0] DOORBELL_ADDR = 8'h10;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      doorbell_pulse <= 1'b0;
      doorbell_value <= 32'd0;
    end else begin
      doorbell_pulse <= 1'b0;
      if (wr_en && (wr_addr == DOORBELL_ADDR)) begin
        doorbell_value <= wr_data;
        doorbell_pulse <= 1'b1;
      end
    end
  end

endmodule
