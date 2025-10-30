// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview


client<llm> Gpt41mini {
  provider "openai-generic"
  options {
    base_url "https://openrouter.ai/api/v1"
    api_key env.OPENROUTER_API_KEY
    model "gpt-4.1-mini"
    temperature 0.0
    headers {
    }
  }
}
