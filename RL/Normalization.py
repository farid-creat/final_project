class NormalizedEnv():
    def __init__(self , action_low , acrion_high):
        self.action_low = action_low
        self.action_high = acrion_high
    def Normalized_to_realspace (self , action):
        act_k = (self.action_high - self.action_low) /2
        act_b = (self.action_high +self.action_low) /2
        return act_k * action +act_b

    def Normalized_action (self,action):
        act_k_inv = 2 / (self.action_high - self.action_low)
        act_b = (self.action_high +self.action_low) /2
        return act_k_inv * (action - act_b)