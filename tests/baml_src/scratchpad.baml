enum Status {
    Accepted @description("Accepted  candidate")
    Refulsed @description("refused  candidate")

}

class Resume2 {
  name string @description("name of the candidate")
  email string
  experience string[]
  skills string[]
  status Status
}

// prefix="If you use this schema correctly

function PrintSchema(  ) -> string | Resume2 {
  client "Gpt41mini" 
  prompt #"
    Print the following text, without interpretation.
    {{ ctx.output_format (prefix="Text: ") }}
  
  "#
}

test TestPrintSchema {
  functions [PrintSchema]
  args {  }
}