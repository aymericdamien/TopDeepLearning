require 'yaml'

module Github
  def self.read(yaml:)
    YAML::load(File.open(yaml, 'r') { |f| f.read })
  end

  def self.md_github_url(repo:)
    "[#{repo.split('/').last}](https://github.com/#{repo})"
  end

  def self.write(content:, yaml:)
    File.open(yaml, 'w') do |f|
      f.write(YAML::dump(content))
    end
  end

  def self.status
    content = read(yaml: DATA_FILE)
    remaining = read(yaml: REPO_LIST).length - content[:resume_index]

    message = <<-STRING
Last Updated : #{content[:last_updated]}
Remaining    : #{remaining}
    STRING

    puts message
  end
end
