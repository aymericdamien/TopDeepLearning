require_relative './utils.rb'

module Github
  LINK_REGEX = /https:\/\/github.com\/(.*?)\)/

  def self.scrape_for_repos(file:)
    text = File
      .open(file, 'r') { |f| f.read }

    text
      .scan(LINK_REGEX)
      .flatten
      .sort
  end

  def self.repo_diff
    from_readme = scrape_for_repos(file: 'README.md')
    from_list = read(yaml: 'repos.yml')
    puts from_readme == from_list.sort ? "No difference" : "Some difference exist"
  end
end
