import torch

def train_grpo(
        policy_model,           # The model being trained
        reference_model,        # Reference model (usually initial SFT model)
        dataset,                # Training questions
        reward_fn,              # Function to calculate rewards
        num_epochs,             # Number of training epochs
        group_size=16,          # Number of outputs to sample per question
        epsilon=0.2,            # PPO clipping parameter
        beta=0.04,              # KL penalty coefficient
        learning_rate=1e-6      # Learning rate
):
    optimizer = Adam(policy_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for question in dataset:
            # Store the current policy as the old policy
            old_policy = copy.deepcopy(policy_model)

            # Sample a group of outputs from the old policy
            group_outputs = []
            for _ in range(group_size):
                output = sample_output(old_policy, question)
                group_outputs.append(output)

            # Calculate rewards for all outputs in the group
            group_rewards = []
            for output in group_outputs:
                reward = reward_fn(question, output)
                group_rewards.append(reward)

            # Normalize rewards to get advantages (no critic model needed)
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8  # Add small epsilon to avoid division by zero
            advantages = [(r - mean_reward) / std_reward for r in group_rewards]

            # Prepare for policy update
            policy_losses = []
            kl_losses = []

            # Calculate losses for each output in the group
            for output, advantage in zip(group_outputs, advantages):
                # For each token position in the output
                for t in range(len(output)):
                    # Get log probabilities from old and current policy
                    current_log_prob = policy_model.log_prob(output[t] | question, output[:t])
                    old_log_prob = old_policy.log_prob(output[t] | question, output[:t])
                    ref_log_prob = reference_model.log_prob(output[t] | question, output[:t])

                    # Calculate probability ratio
                    ratio = torch.exp(current_log_prob - old_log_prob)

                    # Clipped surrogate objective
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1- epsilon, 1 + epsilon) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    policy_losses.append(policy_loss)

                    # KL divergence term (unbiased estimator)
                    ref_ratio = torch.exp(ref_log_prob - current_log_prob)
                    kl_div = ref_ratio - torch.log(ref_ratio) - 1
                    kl_losses.append(kl_div)

            # Calculate the final loss
            policy_loss = torch.mean(torch.stack(policy_losses))
            kl_loss = torch.mean(torch.stack(kl_losses))
            loss = policy_loss + beta * kl_loss

            # Update policy model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy_model