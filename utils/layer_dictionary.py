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

layer_dict = {0: 'down_blocks_0_motion_modules_0',
              1: 'down_blocks_0_motion_modules_1',
              2: 'down_blocks_1_motion_modules_0',
              3: 'down_blocks_1_motion_modules_1',
              4: 'down_blocks_2_motion_modules_0',
              5: 'down_blocks_2_motion_modules_1',
              6: 'down_blocks_3_motion_modules_0',
              7: 'down_blocks_3_motion_modules_1',
              8: 'mid_block_motion_modules_0',
              9: 'up_blocks_0_motion_modules_0',
              10: 'up_blocks_0_motion_modules_1',
              11: 'up_blocks_0_motion_modules_2',
              12: 'up_blocks_1_motion_modules_0',
              13: 'up_blocks_1_motion_modules_1',
              14: 'up_blocks_1_motion_modules_2',
              15: 'up_blocks_2_motion_modules_0',
              16: 'up_blocks_2_motion_modules_1',
              17: 'up_blocks_2_motion_modules_2',
              18: 'up_blocks_3_motion_modules_0',
              19: 'up_blocks_3_motion_modules_1',
              20: 'up_blocks_3_motion_modules_2'}

layer_dict_short = {0: 'down_0_0', 1: 'down_0_1',
                    2: 'down_1_0', 3: 'down_1_1',
                    4: 'down_2_0', 5: 'down_2_1',
                    6: 'down_3_0', 7: 'down_3_1',
                    8: 'mid',
                    9: 'up_0_0', 10: 'up_0_1', 11: 'up_0_2',
                    12: 'up_1_0', 13: 'up_1_1', 14: 'up_1_2',
                    15: 'up_2_0', 16: 'up_2_1', 17: 'up_2_2',
                    18: 'up_3_0', 19: 'up_3_1', 20: 'up_3_2',}


def find_layer_name (skip_layers) :
    target_layers = []
    for layer in skip_layers :
        # find key using value
        target_key = [key for key, value in layer_dict_short.items() if value == layer]
        if len(target_key) != 0 :
            target_layers.append(layer_dict[target_key[0]])
    return target_layers

