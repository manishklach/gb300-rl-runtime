`timescale 1ns/1ps

module tb_rl_runtime_top;

  import desc_pkg::*;

  logic clk;
  logic rst_n;
  desc_t host_desc;
  logic host_desc_valid;
  logic host_desc_ready;
  completion_t host_comp;
  logic host_comp_valid;
  logic host_comp_ready;
  logic doorbell_pulse;

  rl_runtime_top dut (
      .clk(clk),
      .rst_n(rst_n),
      .host_desc(host_desc),
      .host_desc_valid(host_desc_valid),
      .host_comp_ready(host_comp_ready),
      .doorbell_pulse(doorbell_pulse),
      .host_desc_ready(host_desc_ready),
      .host_comp(host_comp),
      .host_comp_valid(host_comp_valid)
  );

  always #5 clk = ~clk;

  task automatic reset_dut;
    begin
      rst_n = 1'b0;
      host_desc = '0;
      host_desc_valid = 1'b0;
      host_comp_ready = 1'b0;
      doorbell_pulse = 1'b0;
      repeat (4) @(posedge clk);
      rst_n = 1'b1;
      @(posedge clk);
    end
  endtask

  task automatic pulse_doorbell;
    begin
      doorbell_pulse = 1'b1;
      @(posedge clk);
      doorbell_pulse = 1'b0;
    end
  endtask

  task automatic submit_desc(input desc_t desc);
    begin
      while (!host_desc_ready)
        @(posedge clk);
      host_desc = desc;
      host_desc_valid = 1'b1;
      @(posedge clk);
      host_desc_valid = 1'b0;
    end
  endtask

  task automatic wait_for_completion(output completion_t comp);
    begin
      host_comp_ready = 1'b1;
      while (!host_comp_valid)
        @(posedge clk);
      comp = host_comp;
      @(posedge clk);
      host_comp_ready = 1'b0;
    end
  endtask

  initial begin
    completion_t comp;
    desc_t desc;

    clk = 1'b0;
    reset_dut();

    // Test 1: short decode completes with DONE.
    desc = '0;
    desc.opcode = DESC_OP_DECODE;
    desc.rollout_id = 16'd7;
    desc.seq_len = 16'd0;
    desc.max_tokens = 16'd10;
    desc.reward_model_id = 16'd3;
    submit_desc(desc);
    pulse_doorbell();
    wait_for_completion(comp);

    if (comp.rollout_id !== 16'd7 ||
        comp.status !== COMP_STATUS_DONE ||
        comp.final_seq_len !== 16'd10) begin
      $display("FAIL: decode completion mismatch");
      $finish(1);
    end

    // Test 2: reward boundary triggers REWARD_NEEDED before max_tokens.
    desc = '0;
    desc.opcode = DESC_OP_DECODE;
    desc.rollout_id = 16'd9;
    desc.seq_len = 16'd0;
    desc.max_tokens = 16'd40;
    desc.reward_model_id = 16'd5;
    submit_desc(desc);
    pulse_doorbell();
    wait_for_completion(comp);

    if (comp.rollout_id !== 16'd9 ||
        comp.status !== COMP_STATUS_REWARD_NEEDED ||
        comp.final_seq_len !== 16'd32) begin
      $display("FAIL: reward boundary completion mismatch");
      $finish(1);
    end

    // Test 3: completion backpressure must not lose the completion.
    desc = '0;
    desc.opcode = DESC_OP_DECODE;
    desc.rollout_id = 16'd11;
    desc.seq_len = 16'd8;
    desc.max_tokens = 16'd12;
    desc.reward_model_id = 16'd1;
    submit_desc(desc);
    pulse_doorbell();

    host_comp_ready = 1'b0;
    repeat (8) @(posedge clk);

    if (!host_comp_valid) begin
      $display("FAIL: completion was not retained under backpressure");
      $finish(1);
    end

    host_comp_ready = 1'b1;
    @(posedge clk);
    comp = host_comp;
    host_comp_ready = 1'b0;

    if (comp.rollout_id !== 16'd11 ||
        comp.status !== COMP_STATUS_DONE ||
        comp.final_seq_len !== 16'd12) begin
      $display("FAIL: backpressure completion mismatch");
      $finish(1);
    end

    $display("PASS");
    $finish(0);
  end

endmodule
