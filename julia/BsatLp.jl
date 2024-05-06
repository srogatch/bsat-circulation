function parse_dimacs(filepath::String)
  # Open the file for reading
  open(filepath, "r") do file
      n_variables = 0
      n_clauses = 0
      clause_list = []

      # Process each line in the file
      for line in eachline(file)
          # Ignore comments and empty lines
          if startswith(line, "c") || isempty(line)
              continue
          elseif startswith(line, "p")
              # Extract problem definition line
              tokens = split(line)
              if length(tokens) != 4 || tokens[1] != "p" || tokens[2] != "cnf"
                  error("Invalid DIMACS header format")
              end
              n_variables = parse(Int, tokens[3])
              n_clauses = parse(Int, tokens[4])
          elseif startswith(line, "%")
            break
          else
              # Process a clause line
              tokens = split(line)
              parsed_clause = parse.(Int, tokens)
              if last(parsed_clause) != 0
                  error("Each clause line must end with a zero.")
              end
              push!(clause_list, parsed_clause[1:end-1])  # Exclude the ending zero
          end
      end

      # Verify that the number of clauses matches the header
      if length(clause_list) != n_clauses
          error("The number of clauses does not match the header.")
      end

      return n_variables, clause_list
  end
end

# Example usage
input_filepath = "bin/uf20-01.cnf"
try
  n_variables, clauses = parse_dimacs(input_filepath)
  println("Number of variables: ", n_variables)
  println("Clauses: ", clauses)
catch e
  println("Error parsing DIMACS file: ", e)
end
