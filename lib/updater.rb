require 'unirest'
require 'yaml'

module Github
  URL = 'https://api.github.com/repos/'
  HEADERS = {'Accept': 'application/vnd.github.preview'}
  REPO_LIST = 'repos.yml'
  DATA_FILE = 'data.yml'

  def self.response(repo:)
    Unirest.get(
      url(repo: repo),
      headers: HEADERS,
    ).body
  end

  def self.url(repo:)
    "#{URL}#{repo}"
  end

  def self.stars(repo:)
    response(repo: repo)['watchers']
  end

  def self.resume_index
    read(yaml: DATA_FILE)[:resume_index]
  end

  def self.update
    repos = read(yaml: REPO_LIST)
    resume_offset = resume_index
    r_index = resume_offset
    partial = []
    partial = read(yaml: DATA_FILE)[:data][0..(r_index - 1)] unless r_index.zero?
    rate_limit_flag = false

    print "Progress: "
    remaining = repos[r_index..-1]
      .map
      .with_index do |repo, index|
      next if rate_limit_flag
      hash = response(repo: repo)
      r_index = index + resume_offset

      if rate_limit?(hash)
        rate_limit_flag = true
        puts "API Rate limit reached for the IP, rerun after an hour to continue."
        next
      end

      print '.'

      {
        repo: repo,
        stars: hash["watchers"],
        description: hash["description"]
      }
    end

    write(content: {
      resume_index: complete?(r_index) ? 0 : r_index,
      last_updated: Time.now,
      data: partial + remaining.compact
    }, yaml: DATA_FILE)
  end

  def self.complete?(index)
    read(yaml: REPO_LIST).length == index + 1
  end

  def self.rate_limit?(hash)
    hash["message"] =~ /API rate limit exceeded/
  end
end
