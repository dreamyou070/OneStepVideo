layers = ['down_blocks_0_motion_modules_0_transformer_blocks_0',
          'down_blocks_0_motion_modules_1_transformer_blocks_0',
          'down_blocks_1_motion_modules_0_transformer_blocks_0',
          'down_blocks_1_motion_modules_1_transformer_blocks_0',
          'down_blocks_2_motion_modules_0_transformer_blocks_0',
          'down_blocks_2_motion_modules_1_transformer_blocks_0',
          'down_blocks_3_motion_modules_0_transformer_blocks_0',
          'down_blocks_3_motion_modules_1_transformer_blocks_0',
          'mid_block_motion_modules_0_transformer_blocks_0',
          'up_blocks_0_motion_modules_0_transformer_blocks_0',
          'up_blocks_0_motion_modules_1_transformer_blocks_0',
          'up_blocks_0_motion_modules_2_transformer_blocks_0',
          'up_blocks_1_motion_modules_0_transformer_blocks_0',
          'up_blocks_1_motion_modules_1_transformer_blocks_0',
          'up_blocks_1_motion_modules_2_transformer_blocks_0',
          'up_blocks_2_motion_modules_0_transformer_blocks_0',
          'up_blocks_2_motion_modules_1_transformer_blocks_0',
          'up_blocks_2_motion_modules_2_transformer_blocks_0',
          'up_blocks_3_motion_modules_0_transformer_blocks_0',
          'up_blocks_3_motion_modules_1_transformer_blocks_0',
          'up_blocks_3_motion_modules_2_transformer_blocks_0', ]

layer_dict = {0: 'down_blocks_0_motion_modules_0_transformer_blocks_0',
              1: 'down_blocks_0_motion_modules_1_transformer_blocks_0',
              2: 'down_blocks_1_motion_modules_0_transformer_blocks_0',
              3: 'down_blocks_1_motion_modules_1_transformer_blocks_0',
              4: 'down_blocks_2_motion_modules_0_transformer_blocks_0',
              5: 'down_blocks_2_motion_modules_1_transformer_blocks_0',
              6: 'down_blocks_3_motion_modules_0_transformer_blocks_0',
              20: 'down_blocks_3_motion_modules_1_transformer_blocks_0',
              7: 'mid_block_motion_modules_0_transformer_blocks_0',
              8: 'up_blocks_0_motion_modules_0_transformer_blocks_0',
              9: 'up_blocks_0_motion_modules_1_transformer_blocks_0',
              10: 'up_blocks_0_motion_modules_2_transformer_blocks_0',
              11: 'up_blocks_1_motion_modules_0_transformer_blocks_0',
              12: 'up_blocks_1_motion_modules_1_transformer_blocks_0',
              13: 'up_blocks_1_motion_modules_2_transformer_blocks_0',
              14: 'up_blocks_2_motion_modules_0_transformer_blocks_0',
              15: 'up_blocks_2_motion_modules_1_transformer_blocks_0',
              16: 'up_blocks_2_motion_modules_2_transformer_blocks_0',
              17: 'up_blocks_3_motion_modules_0_transformer_blocks_0',
              18: 'up_blocks_3_motion_modules_1_transformer_blocks_0',
              19: 'up_blocks_3_motion_modules_2_transformer_blocks_0'}

layer_dict_short = {0: 'down_0_0', 1: 'down_0_1',
              2: 'down_1_0', 3: 'down_1_1',
              4: 'down_2_0', 5: 'down_2_1',
              6: 'down_3_0',20: 'down_3_1',
              7: 'mid',
              8: 'up_0_0', 9: 'up_0_1', 10: 'up_0_2',
              11: 'up_1_0', 12: 'up_1_1', 13: 'up_1_2',
              14: 'up_2_0', 15: 'up_2_1', 16: 'up_2_2',
              17: 'up_3_0', 18: 'up_3_1', 19: 'up_3_2',}


def find_layer_name (skip_layers) :
    target_layers = []
    for layer in skip_layers :
        # find key using value
        target_key = [key for key, value in layer_dict_short.items() if value == layer]
        if len(target_key) != 0 :
            target_layers.append(layer_dict[target_key[0]])
    return target_layers
