    def delegate_work(self, command):
        """Useful to delegate a specific task to a coworker."""
        return self.__execute(command)

    def ask_question(self, command):
        """Useful to ask a question, opinion or take from a coworker."""
        return self.__execute(command)

    def __execute(self, command):
        """Execute the command."""
        try:
            agent, task, context = command.split("|")
        except ValueError:
            return self.i18n.errors("agent_tool_missing_param")

        if not agent or not task or not context:
            return self.i18n.errors("agent_tool_missing_param")

        agent = [
            available_agent
            for available_agent in self.agents
            if available_agent.role == agent
        ]

        if not agent:
            return self.i18n.errors("agent_tool_unexsiting_coworker").format(
                coworkers=", ".join([agent.role for agent in self.agents])
            )

        agent = agent[0]
        return agent.execute_task(task, context)